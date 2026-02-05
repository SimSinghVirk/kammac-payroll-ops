from __future__ import annotations

import io
import json
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
            st.rerun()

    st.header("Validation Overrides")
    skip_pay_elements_validation = st.checkbox("Skip Pay Elements Validation (temporary)")
    ignore_non_numeric_emp_no = st.checkbox("Ignore non-numeric Emp No rows (temporary)")
    ignore_unmapped_zero_activity = st.checkbox("Ignore unmapped with zero activity (temporary)")


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

if auth_required and not st.session_state.logged_in:
    st.subheader("Login Required")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    if st.button("Log in"):
        if username_input == app_username and password_input == app_password:
            st.session_state.logged_in = True
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
                ),
            )
            st.session_state.processed = processed
            st.session_state.exceptions = processed["exceptions"]
            st.session_state.run_status = "VALIDATED"
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

employee_lookup = {}
if processed is not None:
    for _, row in processed["employee_df"].iterrows():
        employee_lookup[str(row.get("employee_id"))] = row

exception_rows = []
for idx, exc in enumerate(exceptions):
    emp = employee_lookup.get(str(exc.employee_id), {})
    name = f"{emp.get('firstname','')} {emp.get('surname','')}".strip()
    exception_rows.append(
        {
            "row_id": idx,
            "employee_id": exc.employee_id,
            "name": name,
            "cost_centre": emp.get("cost_centre", ""),
            "exception_type": exc.exception_type,
            "status": exc.status,
            "dates": _exception_dates(exc.details),
            "summary": _exception_summary(exc.details),
        }
    )

exceptions_df = pd.DataFrame(exception_rows)
if not exceptions_df.empty:
    st.dataframe(
        exceptions_df.drop(columns=["row_id"]),
        use_container_width=True,
        hide_index=True,
    )

if exceptions:
    exc_options = [
        f"{row['employee_id']} | {row['name']} | {row['exception_type']} | {row['status']}"
        for row in exception_rows
    ]
    selection = st.selectbox("Select exception", exc_options)
    exc_index = exc_options.index(selection)
    exc = exceptions[exc_index]

    st.write("Exception details")
    st.json(exc.details)

    st.write("Resolution")
    resolution_action = st.selectbox(
        "Action",
        ["approve_paid", "deduct_unpaid_days", "approve_overtime", "custom_adjustment"],
    )

    extra_fields: dict[str, float] = {}
    if resolution_action == "deduct_unpaid_days":
        extra_fields["deduction_days"] = st.number_input("Deduction days (0.5 steps)", min_value=0.0, step=0.5)
        extra_fields["deduction_hours"] = st.number_input("Deduction hours", min_value=0.0, step=0.25)
    if resolution_action == "approve_overtime":
        extra_fields["overtime_hours"] = st.number_input("Overtime hours", min_value=0.0, step=0.5)
    if resolution_action == "custom_adjustment":
        extra_fields["custom_hours_delta"] = st.number_input(
            "Custom hours adjustment (+ add / - deduct)",
            value=0.0,
            step=0.25,
        )

    comment = st.text_area("Mandatory comment")
    approve = st.button("Save Resolution")

    if approve:
        if not operator_name.strip():
            st.error("Operator name required")
        elif not comment.strip():
            st.error("Comment required")
        else:
            exc.resolution = {"action": resolution_action, **extra_fields}
            exc.prepared_by = operator_name
            exc.prepared_at = datetime.utcnow()
            exc.approved_by = operator_name
            exc.approved_at = datetime.utcnow()
            exc.comments.append(
                {
                    "comment": comment,
                    "author": operator_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            exc.status = "APPROVED"
            st.session_state.audit_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "employee_id": exc.employee_id,
                    "exception_type": exc.exception_type,
                    "action": resolution_action,
                    "details": extra_fields,
                    "comment": comment,
                    "operator": operator_name,
                }
            )
            st.success("Resolution saved")

st.subheader("5. Audit Log")
if st.session_state.audit_log:
    audit_df = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    st.caption("No audit entries yet.")


st.subheader("6. Export")

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
