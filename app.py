from __future__ import annotations

import io
import json
import pickle
import time
from datetime import datetime
import uuid

import pandas as pd
import streamlit as st
import tomllib
from pathlib import Path

try:
    from streamlit.errors import StreamlitSecretNotFoundError
except Exception:  # pragma: no cover
    class StreamlitSecretNotFoundError(Exception):
        pass

from kammac_payroll.constants import PAYROLL_TYPES, SAGE_EXPORT_COLUMNS
from kammac_payroll.export import ExportConfig, build_sage_export
from kammac_payroll.exceptions import ExceptionRecord
from kammac_payroll.google_io import load_sheet_as_df, upload_bytes_to_drive
from kammac_payroll.processing import ProcessingConfig, process_run, normalize_synel_columns
from kammac_payroll.utils import normalize_employee_id
from kammac_payroll.validation import (
    validate_absences,
    validate_mapping,
    validate_pay_elements,
    validate_synel,
)


st.set_page_config(page_title="Kammac Monthly Payroll Ops", layout="wide")

st.title("Kammac Monthly Payroll Ops")

if "run_id" not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())
if "exceptions" not in st.session_state:
    st.session_state.exceptions = []
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "pending_resolutions" not in st.session_state:
    st.session_state.pending_resolutions = {}

# --- Session persistence (login + run state) ---
def _session_token() -> str:
    token = st.session_state.get("session_token")
    if token:
        return token
    qp = st.query_params
    token = qp.get("session")
    if not token:
        token = str(uuid.uuid4())
        qp["session"] = token
        st.rerun()
    st.session_state.session_token = token
    return token


def _sessions_file() -> Path:
    return Path("/tmp/kammac_sessions.json")


def _snapshot_file(token: str) -> Path:
    return Path(f"/tmp/kammac_snapshot_{token}.pkl")


def _load_sessions() -> dict:
    path = _sessions_file()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_sessions(data: dict) -> None:
    _sessions_file().write_text(json.dumps(data))


def _save_snapshot(token: str) -> None:
    snapshot = {
        "run_id": st.session_state.run_id,
        "run_status": st.session_state.run_status,
        "exceptions": st.session_state.exceptions,
        "audit_log": st.session_state.audit_log,
        "processed": st.session_state.get("processed"),
        "export_df": st.session_state.get("export_df"),
    }
    with _snapshot_file(token).open("wb") as fh:
        pickle.dump(snapshot, fh)


def _load_snapshot(token: str) -> None:
    path = _snapshot_file(token)
    if not path.exists():
        return
    try:
        with path.open("rb") as fh:
            snapshot = pickle.load(fh)
        st.session_state.run_id = snapshot.get("run_id", st.session_state.run_id)
        st.session_state.run_status = snapshot.get("run_status", st.session_state.run_status)
        st.session_state.exceptions = snapshot.get("exceptions", st.session_state.exceptions)
        st.session_state.audit_log = snapshot.get("audit_log", st.session_state.audit_log)
        if snapshot.get("processed") is not None:
            st.session_state.processed = snapshot.get("processed")
        if snapshot.get("export_df") is not None:
            st.session_state.export_df = snapshot.get("export_df")
    except Exception:
        pass


token = _session_token()
_load_snapshot(token)
if "run_status" not in st.session_state:
    st.session_state.run_status = "DRAFT"


with st.sidebar:
    st.header("Run Controls")
    payroll_type = st.selectbox("Payroll Type", [""] + PAYROLL_TYPES, index=0)
    period_start = st.date_input("Period Start", value=None)
    period_end = st.date_input("Period End", value=None)

    hourly_threshold = st.number_input("Hourly Threshold", value=100.0, step=1.0)
    variance_tolerance = st.number_input("Variance Tolerance", value=0.5, step=0.25)

    st.header("Google Sheets")
    st.caption("Use one Sheet ID for all tabs or provide separate Sheet IDs below.")
    sheet_id = st.text_input("Sheet ID (all tabs)")
    mapping_sheet_id = st.text_input("Mapping Sheet ID (optional override)")
    absence_sheet_id = st.text_input("Absences Sheet ID (optional override)")
    pay_elements_sheet_id = st.text_input("Pay Elements Sheet ID (optional override)")
    mapping_tab = st.text_input("Mapping Tab", value="AI_PAY_MAPPING_SIM")
    absence_tab = st.text_input("Absences Tab", value="ABSENCES_KAMMAC")
    pay_elements_tab = st.text_input("Pay Elements Tab", value="PAY_ELEMENTS_KAMMAC")

    st.header("Google Drive")
    drive_folder_id = st.text_input("Drive Folder ID")

    operator_name = st.text_input("Operator Name")
    if st.session_state.get("logged_in"):
        if st.button("Log out"):
            st.session_state.logged_in = False
            sessions = _load_sessions()
            if token in sessions:
                sessions.pop(token)
                _save_sessions(sessions)
            st.rerun()

    st.header("Validation Overrides")
    skip_pay_elements_validation = st.checkbox("Skip Pay Elements Validation (temporary)")
    ignore_non_numeric_emp_no = st.checkbox("Ignore non-numeric Emp No rows (temporary)")
    ignore_unmapped_zero_activity = st.checkbox("Ignore unmapped with zero activity (temporary)")
    employee_id_length = st.number_input("Employee ID length (pad)", min_value=0, max_value=12, value=5, step=1)


mapping_df = None
absence_df = None
pay_elements_df = None

def _load_secrets_manual() -> tuple[dict | None, dict]:
    candidates = [
        Path("/Users/simbizz/Documents/New project/.streamlit/secrets.toml"),
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    debug = {"checked": [], "used": None, "error": None}
    for path in candidates:
        debug["checked"].append(str(path))
        if path.exists():
            try:
                data = tomllib.loads(path.read_text())
                debug["used"] = str(path)
                return data, debug
            except Exception:
                debug["error"] = f"Failed to parse {path}"
                continue
    return None, debug


def _get_secret_value(key: str, manual: dict | None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except StreamlitSecretNotFoundError:
        pass
    if manual and key in manual:
        return manual[key]
    return None


credentials_info = None
try:
    if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
        try:
            secrets_value = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
            if isinstance(secrets_value, str) and secrets_value.strip() == "":
                st.error("GOOGLE_SERVICE_ACCOUNT_JSON is empty in Streamlit secrets")
            else:
                credentials_info = json.loads(secrets_value)
        except json.JSONDecodeError:
            st.error("Invalid GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets")
except StreamlitSecretNotFoundError:
    credentials_info = None

manual_debug = None
manual, manual_debug = _load_secrets_manual()
if credentials_info is None:
    if manual and "GOOGLE_SERVICE_ACCOUNT_JSON" in manual:
        try:
            manual_value = manual["GOOGLE_SERVICE_ACCOUNT_JSON"]
            if isinstance(manual_value, str) and manual_value.strip() == "":
                st.error("GOOGLE_SERVICE_ACCOUNT_JSON is empty in secrets.toml")
            else:
                credentials_info = json.loads(manual_value)
        except json.JSONDecodeError:
            st.error("Invalid GOOGLE_SERVICE_ACCOUNT_JSON in secrets.toml")

if credentials_info is None:
    env_json = st.session_state.get("google_sa_json")
    if env_json:
        try:
            credentials_info = json.loads(env_json)
        except json.JSONDecodeError:
            st.error("Invalid Google service account JSON")

app_username = _get_secret_value("APP_USERNAME", manual)
app_password = _get_secret_value("APP_PASSWORD", manual)
auth_required = bool(app_username and app_password)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Restore login from local session store if present
sessions = _load_sessions()
if token in sessions and sessions[token].get("logged_in"):
    st.session_state.logged_in = True

if auth_required and not st.session_state.logged_in:
    st.subheader("Login Required")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    if st.button("Log in"):
        if username_input == app_username and password_input == app_password:
            st.session_state.logged_in = True
            sessions = _load_sessions()
            sessions[token] = {"logged_in": True, "ts": int(time.time())}
            _save_sessions(sessions)
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

st.subheader("1. Load Mapping Data")
if credentials_info is None:
    st.warning("Service account credentials not detected. Add to secrets or paste below.")
    with st.expander("Secrets debug"):
        st.write(manual_debug)
else:
    st.success("Service account credentials detected.")
    with st.expander("Credentials debug"):
        if isinstance(credentials_info, dict):
            st.write({"type": "dict", "keys": list(credentials_info.keys())})
        elif isinstance(credentials_info, str):
            st.write({"type": "str", "length": len(credentials_info), "preview": credentials_info[:50]})
        else:
            st.write({"type": str(type(credentials_info))})

with st.expander("Service Account JSON (if not in secrets)"):
    json_input = st.text_area("Paste service account JSON", height=150)
    if st.button("Use Service Account JSON"):
        st.session_state.google_sa_json = json_input
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

def _load_google_sheets():
    def extract_sheet_id(value: str) -> str:
        if not value:
            return value
        # Accept full Google Sheets URL or raw ID
        if "/spreadsheets/d/" in value:
            try:
                return value.split("/spreadsheets/d/")[1].split("/")[0]
            except Exception:
                return value
        return value

    resolved_mapping_id = extract_sheet_id(mapping_sheet_id or sheet_id)
    resolved_absence_id = extract_sheet_id(absence_sheet_id or sheet_id)
    resolved_pay_elements_id = extract_sheet_id(pay_elements_sheet_id or sheet_id)

    if not resolved_mapping_id or not resolved_absence_id or not resolved_pay_elements_id:
        st.error("Provide a Sheet ID (or all three overrides).")
        return None, None, None
    if credentials_info is None:
        st.error("Service account JSON required")
        return None, None, None

    try:
        creds_payload = credentials_info
        if isinstance(creds_payload, str):
            creds_payload = json.loads(creds_payload)
        if not isinstance(creds_payload, dict):
            st.error(f"Invalid credentials payload type: {type(creds_payload)}")
            return None, None, None

        mapping = load_sheet_as_df(resolved_mapping_id, mapping_tab, creds_payload)
        absence = load_sheet_as_df(resolved_absence_id, absence_tab, creds_payload)
        pay_elements = load_sheet_as_df(resolved_pay_elements_id, pay_elements_tab, creds_payload)
        return mapping, absence, pay_elements
    except Exception as exc:
        st.error(
            "Failed to load Google Sheets. "
            "Check Sheet IDs, tab names, and sharing. "
            f"Error: {exc}"
        )
        st.caption(
            "Sheet IDs used: "
            f"mapping={resolved_mapping_id}, absences={resolved_absence_id}, pay_elements={resolved_pay_elements_id}"
        )
        st.caption(
            "Tabs used: "
            f"mapping={mapping_tab}, absences={absence_tab}, pay_elements={pay_elements_tab}"
        )
        return None, None, None


current_config = {
    "sheet_id": sheet_id,
    "mapping_sheet_id": mapping_sheet_id,
    "absence_sheet_id": absence_sheet_id,
    "pay_elements_sheet_id": pay_elements_sheet_id,
    "mapping_tab": mapping_tab,
    "absence_tab": absence_tab,
    "pay_elements_tab": pay_elements_tab,
}

if "last_sheet_config" not in st.session_state:
    st.session_state.last_sheet_config = None

auto_load_ready = any(
    [
        sheet_id,
        mapping_sheet_id,
        absence_sheet_id,
        pay_elements_sheet_id,
    ]
)

if "mapping_df" not in st.session_state:
    st.session_state.mapping_df = None
if "absence_df" not in st.session_state:
    st.session_state.absence_df = None
if "pay_elements_df" not in st.session_state:
    st.session_state.pay_elements_df = None

if auto_load_ready and st.session_state.last_sheet_config != current_config:
    mapping_df, absence_df, pay_elements_df = _load_google_sheets()
    st.session_state.mapping_df = mapping_df
    st.session_state.absence_df = absence_df
    st.session_state.pay_elements_df = pay_elements_df
    st.session_state.last_sheet_config = current_config
    st.session_state.last_sheet_load_at = datetime.utcnow().isoformat()
else:
    mapping_df = st.session_state.mapping_df
    absence_df = st.session_state.absence_df
    pay_elements_df = st.session_state.pay_elements_df

st.caption("Mapping data auto-loads when Sheet IDs change.")
if auto_load_ready:
    last_load = st.session_state.get("last_sheet_load_at")
    if last_load:
        st.caption(f"Last sheet load (UTC): {last_load}")

if mapping_df is not None:
    st.success(f"Loaded mapping rows: {len(mapping_df)}")
if absence_df is not None:
    st.success(f"Loaded absences rows: {len(absence_df)}")
if pay_elements_df is not None:
    st.success(f"Loaded pay elements rows: {len(pay_elements_df)}")


st.subheader("2. Upload Synel Monthly File")

synel_file = st.file_uploader("Upload Synel CSV or XLSX", type=["csv", "xlsx"])

synel_df = None
synel_override_warnings: list[str] = []
if synel_file is not None:
    try:
        def read_synel_with_header_detection(file_obj):
            name = file_obj.name.lower()
            data = file_obj.read()
            file_obj.seek(0)

            def read_with_header(header_idx):
                buf = io.BytesIO(data)
                if name.endswith(".xlsx"):
                    return pd.read_excel(buf, header=header_idx)
                return pd.read_csv(buf, header=header_idx)

            def read_no_header():
                buf = io.BytesIO(data)
                if name.endswith(".xlsx"):
                    return pd.read_excel(buf, header=None)
                return pd.read_csv(buf, header=None)

            df = read_with_header(0)
            normalized = normalize_synel_columns(df.copy())
            required = {"Emp No", "Date", "IN_1", "OUT_1", "IN_2", "OUT_2", "ABS_HALF_DAY_1", "ABS_HALF_DAY_2"}
            if required.issubset(set(normalized.columns)):
                return df

            raw = read_no_header()
            header_row = None
            for idx in range(min(10, len(raw))):
                row_values = [str(v).strip() for v in raw.iloc[idx].tolist()]
                if "Emp No" in row_values and "Date" in row_values:
                    header_row = idx
                    break
            if header_row is not None:
                return read_with_header(header_row)
            return df

        synel_df = read_synel_with_header_detection(synel_file)
        if ignore_non_numeric_emp_no and synel_df is not None:
            def _is_valid_emp(value):
                return normalize_employee_id(value).valid

            before = len(synel_df)
            synel_df = synel_df[synel_df["Emp No"].apply(_is_valid_emp)].copy()
            after = len(synel_df)
            if after < before:
                synel_override_warnings.append(
                    f"Ignored {before - after} rows with non-numeric Emp No (temporary override)."
                )
    except Exception as exc:
        st.error(f"Unable to read Synel file: {exc}")

if synel_df is not None:
    st.write(f"Synel rows loaded: {len(synel_df)}")


st.subheader("3. Validate & Process")

blocking_errors: list[str] = []
warnings: list[str] = []
warnings.extend(synel_override_warnings)

if period_start is None or period_end is None:
    blocking_errors.append("Period Start and Period End are required.")
if payroll_type == "":
    blocking_errors.append("Payroll Type is required.")

if mapping_df is not None:
    mapping_validation = validate_mapping(mapping_df)
    blocking_errors.extend(mapping_validation.blocking)
    warnings.extend(mapping_validation.warnings)
if absence_df is not None:
    absence_validation = validate_absences(absence_df)
    blocking_errors.extend(absence_validation.blocking)
    warnings.extend(absence_validation.warnings)
if pay_elements_df is not None and not skip_pay_elements_validation:
    pay_validation = validate_pay_elements(pay_elements_df)
    blocking_errors.extend(pay_validation.blocking)
    warnings.extend(pay_validation.warnings)
elif pay_elements_df is not None and skip_pay_elements_validation:
    warnings.append("Pay elements validation skipped (temporary override).")
if synel_df is not None:
    synel_validation = validate_synel(normalize_synel_columns(synel_df.copy()))
    blocking_errors.extend(synel_validation.blocking)
    warnings.extend(synel_validation.warnings)

if blocking_errors:
    st.error("Blocking Issues")
    for issue in blocking_errors:
        st.write(f"- {issue}")

if warnings:
    st.warning("Warnings")
    for issue in warnings:
        st.write(f"- {issue}")

can_process = not blocking_errors and all(
    df is not None for df in [mapping_df, absence_df, pay_elements_df, synel_df]
)

processed = None
if can_process:
    if st.button("Run Processing"):
        try:
            processed = process_run(
                mapping_df,
                absence_df,
                pay_elements_df,
                synel_df,
                ProcessingConfig(
                    payroll_type=payroll_type,
                    period_start=pd.Timestamp(period_start),
                    period_end=pd.Timestamp(period_end),
                    hourly_threshold=float(hourly_threshold),
                    variance_tolerance=float(variance_tolerance),
                    ignore_unmapped_zero_activity=ignore_unmapped_zero_activity,
                    employee_id_length=int(employee_id_length) if employee_id_length else None,
                ),
            )
            st.session_state.processed = processed
            st.session_state.exceptions = processed["exceptions"]
            st.session_state.run_status = "VALIDATED"
            _save_snapshot(token)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

if "processed" in st.session_state:
    processed = st.session_state.processed

if processed is not None:
    st.subheader("Run Summary")
    st.write(f"Unmapped employees: {len(processed['unmapped_employees'])}")
    if processed["unmapped_employees"]:
        st.write(processed["unmapped_employees"])

    if processed.get("blocking"):
        st.error("Processing Blocking Issues")
        for issue in processed["blocking"]:
            st.write(f"- {issue}")

    st.dataframe(processed["employee_df"], use_container_width=True)


st.subheader("4. Exceptions")
exceptions: list[ExceptionRecord] = st.session_state.exceptions
open_exceptions = [e for e in exceptions if not e.is_resolved()]

st.write(f"Open exceptions: {len(open_exceptions)}")

def _exception_dates(details: dict) -> str:
    if not details:
        return ""
    if "date" in details:
        return str(details.get("date"))
    if "rows" in details:
        dates = [str(r.get("date")) for r in details.get("rows", []) if r.get("date")]
        return ", ".join(sorted(set(dates))[:5])
    return ""

def _exception_summary(details: dict) -> str:
    if not details:
        return ""
    if "actual_hours" in details and "expected_hours" in details:
        return f"actual={details['actual_hours']}, expected={details['expected_hours']}"
    if "reason" in details:
        return str(details.get("reason"))
    if "rows" in details:
        codes = sorted(set(r.get("code") for r in details.get("rows", []) if r.get("code")))
        return f"codes={', '.join([str(c) for c in codes if c])}"
    return ""

def _truncate(text: str, limit: int = 40) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"

employee_lookup = {}
if processed is not None:
    for _, row in processed["employee_df"].iterrows():
        employee_lookup[str(row.get("employee_id"))] = row

exception_rows = []
exception_row_by_id = {}
for idx, exc in enumerate(exceptions):
    emp = employee_lookup.get(str(exc.employee_id), {})
    name = f"{emp.get('firstname','')} {emp.get('surname','')}".strip()
    row = {
        "row_id": idx,
        "exception_id": exc.exception_id,
        "employee_id": exc.employee_id,
        "name": name,
        "cost_centre": emp.get("cost_centre", ""),
        "exception_type": exc.exception_type,
        "status": exc.status,
        "dates": _exception_dates(exc.details),
        "summary": _exception_summary(exc.details),
    }
    exception_rows.append(row)
    exception_row_by_id[exc.exception_id] = row

def _allowed_actions(exc_type: str) -> list[str]:
    if exc_type == "OVER_HOURS_SALARIED":
        return ["approve_paid", "approve_overtime", "custom_adjustment"]
    if exc_type in ["UNDER_HOURS_SALARIED", "NO_PUNCHES_SALARIED", "MISSING_PUNCH_DAY"]:
        return ["approve_paid", "deduct_unpaid_days", "custom_adjustment"]
    if exc_type == "ABSENCE_CODE_UNKNOWN":
        return ["acknowledge"]
    return ["approve_paid", "custom_adjustment"]

if not exceptions:
    st.caption("No exceptions.")
else:
    st.caption("Use the table to review. Expand a row to edit. Click Apply All Approvals to save.")
    show_open_only = st.checkbox("Show open exceptions only", value=True)
    per_page = st.selectbox("Exceptions per page", [10, 20, 30, 50], index=0)

    visible_exceptions = open_exceptions if show_open_only else exceptions
    total_pages = max(1, (len(visible_exceptions) + per_page - 1) // per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    visible_exceptions = visible_exceptions[start_idx:end_idx]

    if not visible_exceptions:
        st.caption("No exceptions in this view.")

    with st.form("exception_approvals_form", clear_on_submit=False):
        header = st.columns([1.2, 2.4, 2.2, 2.2, 3.2, 1.2])
        header[0].markdown("**Employee**")
        header[1].markdown("**Name**")
        header[2].markdown("**Cost Centre**")
        header[3].markdown("**Type**")
        header[4].markdown("**Dates / Summary**")
        header[5].markdown("**Status**")

        pending_updates: dict[str, dict] = {}

        for exc in visible_exceptions:
            row = exception_row_by_id.get(exc.exception_id)
            if row is None:
                row = {
                    "exception_id": exc.exception_id,
                    "employee_id": exc.employee_id,
                    "name": "",
                    "cost_centre": "",
                    "exception_type": exc.exception_type,
                    "status": exc.status,
                    "dates": _exception_dates(exc.details),
                    "summary": _exception_summary(exc.details),
                }

            info = st.columns([1.2, 2.4, 2.2, 2.2, 3.2, 1.2])
            info[0].write(row["employee_id"])
            info[1].write(_truncate(row["name"], 28))
            info[2].write(_truncate(row["cost_centre"], 24))
            info[3].write(_truncate(row["exception_type"], 28))
            summary_text = f"{row['dates']} {row['summary']}".strip()
            info[4].write(_truncate(summary_text, 60))
            info[5].write(row["status"])

            with st.expander(f"Edit {row['employee_id']} | {row['exception_type']}", expanded=False):
                actions = _allowed_actions(exc.exception_type)
                selected_action = st.selectbox(
                    "Action",
                    actions,
                    key=f"action_{exc.exception_id}",
                )

                extra_fields: dict[str, float] = {}
                if selected_action == "deduct_unpaid_days":
                    extra_fields["deduction_days"] = st.number_input(
                        "Deduction days",
                        min_value=0.0,
                        step=0.5,
                        key=f"ded_days_{exc.exception_id}",
                    )
                    extra_fields["deduction_hours"] = st.number_input(
                        "Deduction hours",
                        min_value=0.0,
                        step=0.25,
                        key=f"ded_hours_{exc.exception_id}",
                    )
                elif selected_action == "approve_overtime":
                    extra_fields["overtime_hours"] = st.number_input(
                        "Overtime hours",
                        min_value=0.0,
                        step=0.5,
                        key=f"ot_hours_{exc.exception_id}",
                    )
                elif selected_action == "custom_adjustment":
                    extra_fields["custom_hours_delta"] = st.number_input(
                        "Custom hours (+/-)",
                        value=0.0,
                        step=0.25,
                        key=f"custom_hours_{exc.exception_id}",
                    )

                comment = st.text_input(
                    "Comment (optional)",
                    key=f"comment_{exc.exception_id}",
                )

                pending_updates[exc.exception_id] = {
                    "action": selected_action,
                    "details": extra_fields,
                    "comment": comment,
                }

                st.markdown(
                    f"**Details:** {row['summary']}\n\n"
                    f"- Dates: {row['dates']}\n"
                    f"- Cost Centre: {row['cost_centre']}\n"
                )
                st.json(exc.details)
            st.markdown("---")

        apply_clicked = st.form_submit_button("Apply All Approvals")

    if apply_clicked:
        effective_operator = operator_name.strip() if operator_name.strip() else (app_username or "unknown")
        applied = 0
        for exc in exceptions:
            pending = pending_updates.get(exc.exception_id)
            if not pending:
                continue
            exc.resolution = {"action": pending["action"], **pending["details"]}
            exc.prepared_by = effective_operator
            exc.prepared_at = datetime.utcnow()
            exc.approved_by = effective_operator
            exc.approved_at = datetime.utcnow()
            if pending.get("comment"):
                exc.comments.append(
                    {
                        "comment": pending["comment"],
                        "author": effective_operator,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            exc.status = "APPROVED"
            st.session_state.audit_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "employee_id": exc.employee_id,
                    "exception_type": exc.exception_type,
                    "action": pending["action"],
                    "details": pending["details"],
                    "comment": pending["comment"],
                    "operator": effective_operator,
                }
            )
            applied += 1
        if applied:
            st.success(f"Applied approvals: {applied}")
            _save_snapshot(token)
            st.rerun()

    if processed is not None:
        # Optional diagnostics (hidden by default)
        with st.expander("Diagnostics (optional)"):
            missing_rows = [e for e in exceptions if e.exception_type == "MISSING_PUNCH_DAY"]
            if missing_rows:
                diag_rows = []
                synel_period = processed.get("synel_period")
                if synel_period is not None:
                    for exc in missing_rows:
                        date_val = exc.details.get("date")
                        emp = str(exc.employee_id)
                        matches = synel_period[
                            (synel_period["Emp No"].astype(str).str.strip() == emp)
                            & (synel_period["Date"] == date_val)
                        ]
                        if matches.empty:
                            diag_rows.append(
                                {
                                    "employee_id": emp,
                                    "date": date_val,
                                    "IN_1": "",
                                    "OUT_1": "",
                                    "IN_2": "",
                                    "OUT_2": "",
                                    "ABS_1": "",
                                    "ABS_2": "",
                                    "note": "No row found",
                                }
                            )
                        else:
                            for _, row in matches.iterrows():
                                diag_rows.append(
                                    {
                                        "employee_id": emp,
                                        "date": row.get("Date"),
                                        "IN_1": row.get("IN_1", ""),
                                        "OUT_1": row.get("OUT_1", ""),
                                        "IN_2": row.get("IN_2", ""),
                                        "OUT_2": row.get("OUT_2", ""),
                                        "ABS_1": row.get("ABS_HALF_DAY_1", ""),
                                        "ABS_2": row.get("ABS_HALF_DAY_2", ""),
                                        "note": "",
                                    }
                                )
                    st.caption("Missing punch rows with exact IN/OUT values.")
                    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

st.subheader("5. Audit Log")
if st.session_state.audit_log:
    audit_df = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    st.caption("No audit entries yet.")

st.subheader("6. Final Review")
if processed is not None:
    exceptions_by_emp: dict[str, list[ExceptionRecord]] = {}
    for exc in exceptions:
        exceptions_by_emp.setdefault(str(exc.employee_id), []).append(exc)

    # Build a stable list of absence codes from mapping
    absence_codes = []
    if absence_df is not None and "Short Name" in absence_df.columns:
        absence_codes = (
            absence_df["Short Name"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        absence_codes = [c for c in absence_codes if c]
    absence_codes = sorted(set(absence_codes))

    review_rows = []
    for _, row in processed["employee_df"].iterrows():
        employee_id = str(row.get("employee_id"))
        if employee_id_length:
            employee_id = employee_id.zfill(int(employee_id_length))
        pay_basis = row.get("pay_basis")
        weekly_hours = row.get("weekly_hours") or 0.0
        standard_hours = row.get("standard_monthly_hours") or 0.0
        annual_salary = row.get("basic_pay_value") or 0.0
        hourly_rate = 0.0
        if pay_basis == "HOURLY":
            hourly_rate = annual_salary
        elif weekly_hours:
            hourly_rate = annual_salary / (weekly_hours * 52.0)

        deduction_days = 0.0
        deduction_hours = 0.0
        custom_hours_delta = 0.0
        overtime_hours = 0.0
        for exc in exceptions_by_emp.get(employee_id, []):
            if not exc.is_resolved():
                continue
            resolution = exc.resolution or {}
            action = resolution.get("action")
            if action == "deduct_unpaid_days":
                deduction_days += float(resolution.get("deduction_days") or 0.0)
                deduction_hours += float(resolution.get("deduction_hours") or 0.0)
            if action == "approve_overtime":
                overtime_hours += float(resolution.get("overtime_hours") or 0.0)
            if action == "custom_adjustment":
                custom_hours_delta += float(resolution.get("custom_hours_delta") or 0.0)

        hours_per_day = (weekly_hours / 5.0) if weekly_hours else 0.0
        final_base_hours = standard_hours - (deduction_days * hours_per_day) - deduction_hours + custom_hours_delta
        if final_base_hours < 0:
            final_base_hours = 0.0

        base_monthly_pay = annual_salary / 12.0 if pay_basis == "SALARIED" else hourly_rate * final_base_hours
        deduction_money = (deduction_days * hours_per_day + deduction_hours) * hourly_rate
        custom_money = custom_hours_delta * hourly_rate
        overtime_money = overtime_hours * hourly_rate
        # Allowances
        car_allowance = row.get("car_allowance") or 0.0
        period_days = 0
        if processed is not None:
            synel_period = processed.get("synel_period")
            if synel_period is not None and not synel_period.empty:
                period_days = synel_period["Date"].nunique()
        weeks_in_period = int(period_days / 7.0 + 0.9999) if period_days else 0
        fire_marshal_weekly = row.get("fm_fa_weekly") or 0.0
        fire_marshal_pay = fire_marshal_weekly * weeks_in_period

        total_money = (
            base_monthly_pay
            - deduction_money
            + custom_money
            + overtime_money
            + car_allowance
            + fire_marshal_pay
        )

        row_out = {
                "Employee Id": employee_id,
                "Name": f"{row.get('firstname','')} {row.get('surname','')}".strip(),
                "Cost Centre": row.get("cost_centre", ""),
                "Pay Basis": pay_basis,
                "Annual Salary": round(annual_salary, 2) if pay_basis == "SALARIED" else "",
                "Base Monthly Hours": round(standard_hours, 2),
                "Final Base Hours": round(final_base_hours, 2),
                "Deduction Days": round(deduction_days, 2),
                "Deduction Hours": round(deduction_hours, 2),
                "Overtime Hours": round(overtime_hours, 2),
                "Custom Hours Delta": round(custom_hours_delta, 2),
                "Hourly Rate (est)": round(hourly_rate, 4),
                "Base Monthly Pay": round(base_monthly_pay, 2),
                "Deduction Money": round(deduction_money, 2),
                "Custom Money": round(custom_money, 2),
                "Overtime Money": round(overtime_money, 2),
                "Car Allowance": round(car_allowance, 2),
                "FM/FA Weekly": round(fire_marshal_weekly, 2),
                "FM/FA Weeks": round(weeks_in_period, 3),
                "FM/FA Pay": round(fire_marshal_pay, 2),
                "Total Pay (est)": round(total_money, 2),
            }

        # Add absence code breakdown columns
        abs_map = row.get("absence_days_by_code") or {}
        for code in absence_codes:
            days = abs_map.get(code, 0.0)
            row_out[f"Absence {code} Days"] = round(days, 2)
            # Estimate pay impact per code
            if pay_basis == "SALARIED":
                daily_rate = row.get("daily_rate") or 0.0
                row_out[f"Absence {code} Pay"] = round(days * daily_rate, 2)
            else:
                row_out[f"Absence {code} Pay"] = round(days * hours_per_day * hourly_rate, 2)

        review_rows.append(row_out)

    review_df = pd.DataFrame(review_rows)
    st.dataframe(review_df, use_container_width=True, hide_index=True)
    review_csv = review_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Final Review CSV",
        data=review_csv,
        file_name=f"FINAL_REVIEW_{st.session_state.run_id}.csv",
        mime="text/csv",
    )
else:
    st.caption("Run processing to generate the final review report.")


st.subheader("7. Export")

all_resolved = all(exc.is_resolved() for exc in exceptions)
processing_blocking = processed.get("blocking") if processed else []
export_ready = can_process and processed is not None and all_resolved and not processing_blocking

if not all_resolved and exceptions:
    st.warning("Export disabled until all exceptions are approved.")

if export_ready:
    if st.button("Generate Sage Export"):
        try:
            export_df = build_sage_export(
                processed["employee_df"],
                exceptions,
                pay_elements_df,
                ExportConfig(payroll_type=payroll_type),
            )
            st.session_state.export_df = export_df
            st.session_state.run_status = "READY_TO_EXPORT"
            _save_snapshot(token)
        except Exception as exc:
            st.error(f"Export failed: {exc}")

if "export_df" in st.session_state:
    export_df = st.session_state.export_df
    st.dataframe(export_df, use_container_width=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Sage Export CSV",
        data=csv_bytes,
        file_name=f"SAGE_EXPORT_{st.session_state.run_id}.csv",
        mime="text/csv",
    )

    if drive_folder_id and credentials_info:
        if st.button("Upload Export Pack to Drive"):
            try:
                file_id = upload_bytes_to_drive(
                    drive_folder_id,
                    f"SAGE_EXPORT_{st.session_state.run_id}.csv",
                    csv_bytes,
                    "text/csv",
                    credentials_info,
                )
                st.success(f"Uploaded to Drive file ID: {file_id}")
                st.session_state.run_status = "EXPORTED"
            except Exception as exc:
                st.error(f"Drive upload failed: {exc}")

st.caption(f"Run ID: {st.session_state.run_id} | Status: {st.session_state.run_status}")
