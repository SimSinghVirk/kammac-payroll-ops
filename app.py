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
from kammac_payroll.utils import normalize_employee_id, coerce_numeric, parse_time_to_hours, safe_str
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
if "manual_adjustments" not in st.session_state:
    st.session_state.manual_adjustments = []
if "date_adjustments" not in st.session_state:
    st.session_state.date_adjustments = []

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
        "date_adjustments": st.session_state.get("date_adjustments", []),
        "manual_adjustments": st.session_state.get("manual_adjustments", []),
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
        if snapshot.get("date_adjustments") is not None:
            st.session_state.date_adjustments = snapshot.get("date_adjustments")
        if snapshot.get("manual_adjustments") is not None:
            st.session_state.manual_adjustments = snapshot.get("manual_adjustments")
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    safe_df = df.copy()
    for col in safe_df.columns:
        series = safe_df[col]
        if series.apply(lambda v: isinstance(v, (dict, list, tuple, set))).any():
            safe_df[col] = series.apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple, set)) else v
            )
    return safe_df


def _row_worked_hours_from_punches(row: pd.Series) -> float:
    in1 = parse_time_to_hours(row.get("IN_1"))
    out1 = parse_time_to_hours(row.get("OUT_1"))
    in2 = parse_time_to_hours(row.get("IN_2"))
    out2 = parse_time_to_hours(row.get("OUT_2"))

    total = 0.0
    if in1 is not None and out1 is not None and out1 >= in1:
        total += out1 - in1
    if in2 is not None and out2 is not None and out2 >= in2:
        total += out2 - in2
    return total


def _row_overtime_hours(row: pd.Series) -> float:
    time_half = coerce_numeric(row.get("TIME & A 1/2")) or 0.0
    overtime = coerce_numeric(row.get("OVERTIME")) or 0.0
    return time_half if time_half > 0 else overtime


def _sum_date_adjustments(employee_id: str) -> tuple[float, dict[str, float]]:
    total = 0.0
    by_bucket: dict[str, float] = {}
    for adj in st.session_state.get("date_adjustments", []):
        if str(adj.get("employee_id")) != str(employee_id):
            continue
        hours = float(adj.get("hours") or 0.0)
        bucket = adj.get("bucket", "BASIC")
        total += hours
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + hours
    return total, by_bucket


def _compute_employee_totals(
    row: pd.Series,
    exceptions_by_emp: dict[str, list[ExceptionRecord]],
    absence_paid_map: dict[str, bool],
    processed: dict[str, Any] | None,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    employee_id = str(row.get("employee_id"))
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
    custom_money_delta = 0.0
    custom_by_code_hours: dict[str, float] = {}
    custom_by_code_money: dict[str, float] = {}
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
            adj_type = resolution.get("custom_adjustment_type", "hours")
            adj_amount = float(resolution.get("custom_adjustment_amount") or 0.0)
            bucket = resolution.get("custom_adjustment_bucket", "BASIC")
            if adj_type == "money":
                custom_money_delta += adj_amount
                custom_by_code_money[bucket] = custom_by_code_money.get(bucket, 0.0) + adj_amount
            else:
                custom_hours_delta += adj_amount
                custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + adj_amount

    for adj in st.session_state.manual_adjustments:
        if str(adj.get("employee_id")) != employee_id:
            continue
        adj_type = adj.get("type", "hours")
        adj_amount = float(adj.get("amount") or 0.0)
        bucket = adj.get("bucket", "BASIC")
        if adj_type == "money":
            custom_money_delta += adj_amount
            custom_by_code_money[bucket] = custom_by_code_money.get(bucket, 0.0) + adj_amount
        else:
            custom_hours_delta += adj_amount
            custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + adj_amount

    date_hours_delta, date_hours_by_bucket = _sum_date_adjustments(employee_id)
    custom_hours_delta += date_hours_delta
    for bucket, hours in date_hours_by_bucket.items():
        custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + hours

    if extra:
        deduction_days += float(extra.get("deduction_days") or 0.0)
        deduction_hours += float(extra.get("deduction_hours") or 0.0)
        overtime_hours += float(extra.get("overtime_hours") or 0.0)
        custom_hours_delta += float(extra.get("custom_hours_delta") or 0.0)
        custom_money_delta += float(extra.get("custom_money_delta") or 0.0)
        for bucket, hours in (extra.get("custom_by_code_hours") or {}).items():
            custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + float(hours or 0.0)
        for bucket, money in (extra.get("custom_by_code_money") or {}).items():
            custom_by_code_money[bucket] = custom_by_code_money.get(bucket, 0.0) + float(money or 0.0)

    hours_per_day = (weekly_hours / 5.0) if weekly_hours else 8.0
    final_base_hours = standard_hours - (deduction_days * hours_per_day) - deduction_hours + custom_hours_delta
    if final_base_hours < 0:
        final_base_hours = 0.0

    synel_overtime_hours = row.get("overtime_hours_synel") or 0.0
    worked_hours = row.get("worked_hours") or 0.0
    regular_hours = max((worked_hours or 0.0), 0.0)

    abs_map = row.get("absence_days_by_code") or {}
    abs_hours_map = row.get("absence_hours_by_code") or {}

    if pay_basis == "SALARIED":
        base_monthly_pay = annual_salary / 12.0
        overtime_money = overtime_hours * hourly_rate
        paid_abs_hours = 0.0
        unpaid_abs_hours = 0.0
    else:
        paid_abs_hours = sum(
            hours for code, hours in abs_hours_map.items() if absence_paid_map.get(code, False)
        )
        unpaid_abs_hours = sum(
            hours for code, hours in abs_hours_map.items() if not absence_paid_map.get(code, False)
        )
        base_monthly_pay = (regular_hours + paid_abs_hours) * hourly_rate
        overtime_money = synel_overtime_hours * hourly_rate * 1.5

    deduction_money = (deduction_days * hours_per_day + deduction_hours) * hourly_rate
    custom_money = (custom_hours_delta * hourly_rate) + custom_money_delta

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

    return {
        "total_money": total_money,
        "base_monthly_pay": base_monthly_pay,
        "deduction_money": deduction_money,
        "custom_money": custom_money,
        "overtime_money": overtime_money,
        "regular_hours": regular_hours,
        "paid_absence_hours": paid_abs_hours,
        "unpaid_absence_hours": unpaid_abs_hours,
        "custom_by_code_hours": custom_by_code_hours,
        "custom_by_code_money": custom_by_code_money,
    }


def _build_daily_breakdown(
    employee_id: str,
    processed: dict[str, Any] | None,
    period_start: datetime | None,
    period_end: datetime | None,
    absence_codes: list[str],
    absence_paid_map: dict[str, bool],
) -> pd.DataFrame:
    if processed is None or period_start is None or period_end is None:
        return pd.DataFrame()

    synel_period = processed.get("synel_period")
    if synel_period is None or synel_period.empty:
        return pd.DataFrame()

    emp_id = str(employee_id)
    rows = synel_period[synel_period["Emp No"].astype(str).str.strip() == emp_id]

    date_index = pd.date_range(start=pd.Timestamp(period_start), end=pd.Timestamp(period_end), freq="D")
    daily: dict[pd.Timestamp, dict[str, Any]] = {}
    for d in date_index:
        row = {
            "Date": d.date().isoformat(),
            "Regular Hours": 0.0,
            "Overtime Hours": 0.0,
            "Holiday Pay Hours": 0.0,
            "Bank Holiday Pay Hours": 0.0,
            "Sickness Hours": 0.0,
            "Paid Absence Hours": 0.0,
            "Unpaid Absence Hours": 0.0,
            "Adjustment Hours": 0.0,
        }
        for code in absence_codes:
            row[f"Absence {code} Hours"] = 0.0
        daily[d.normalize()] = row

    for _, row in rows.iterrows():
        date_val = row.get("Date")
        if pd.isna(date_val):
            continue
        key = pd.Timestamp(date_val).normalize()
        if key not in daily:
            continue
        target = daily[key]

        codes = []
        code1 = safe_str(row.get("ABS_HALF_DAY_1")).upper()
        code2 = safe_str(row.get("ABS_HALF_DAY_2")).upper()
        if code1:
            codes.append(code1)
        if code2:
            codes.append(code2)

        basic_hours = coerce_numeric(row.get("BASIC PAY")) or 0.0
        row_total = coerce_numeric(row.get("_row_total_hours")) or 0.0

        if not codes:
            if basic_hours > 0:
                target["Regular Hours"] += basic_hours
            else:
                if row_total <= 0:
                    target["Regular Hours"] += _row_worked_hours_from_punches(row)
        else:
            if basic_hours > 0:
                per_code = basic_hours / len(codes)
                for code in codes:
                    col = f"Absence {code} Hours"
                    if col in target:
                        target[col] += per_code

        target["Overtime Hours"] += _row_overtime_hours(row)
        target["Holiday Pay Hours"] += coerce_numeric(row.get("HOLIDAY PAY")) or 0.0
        target["Bank Holiday Pay Hours"] += coerce_numeric(row.get("BANK HOLIDAY PAY")) or 0.0
        target["Sickness Hours"] += coerce_numeric(row.get("SUP - Sickness")) or 0.0

    for adj in st.session_state.get("date_adjustments", []):
        if str(adj.get("employee_id")) != emp_id:
            continue
        adj_date = adj.get("date")
        if not adj_date:
            continue
        try:
            key = pd.Timestamp(adj_date).normalize()
        except Exception:
            continue
        if key in daily:
            daily[key]["Adjustment Hours"] += float(adj.get("hours") or 0.0)

    for row in daily.values():
        paid = 0.0
        unpaid = 0.0
        for code in absence_codes:
            hours = row.get(f"Absence {code} Hours", 0.0)
            if absence_paid_map.get(code, False):
                paid += hours
            else:
                unpaid += hours
        row["Paid Absence Hours"] = paid
        row["Unpaid Absence Hours"] = unpaid

    return pd.DataFrame(list(daily.values()))


token = _session_token()
_load_snapshot(token)
if "run_status" not in st.session_state:
    st.session_state.run_status = "DRAFT"


with st.sidebar:
    st.header("Run Controls")
    view_mode = st.selectbox("View", ["Payroll Run", "Employee Lookup"])
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
    excluded_locations = []
    mapping_preview = st.session_state.get("mapping_df")
    if mapping_preview is not None:
        loc_df = mapping_preview.copy()
        if "Payroll Type" not in loc_df.columns and "COST CENTRE" in loc_df.columns:
            def _derive_pt(value):
                text = str(value).strip().lower()
                if text.startswith("1") or "direct monthly" in text:
                    return "direct_monthly"
                if text.startswith("2") or "admin monthly" in text:
                    return "admin_monthly"
                return "unknown"
            loc_df["Payroll Type"] = loc_df["COST CENTRE"].apply(_derive_pt)
        if payroll_type:
            loc_df = loc_df[loc_df["Payroll Type"] == payroll_type]
        available_locations = sorted(loc_df["Location"].astype(str).str.strip().dropna().unique().tolist())
        excluded_locations = st.multiselect("Exclude Locations", available_locations)
    else:
        st.caption("Load mapping to select locations for exclusion.")


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
    st.warning("Service account credentials not detected. Add to secrets.")
else:
    st.success("Service account credentials detected.")

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
            exclude_locations_list = excluded_locations if isinstance(excluded_locations, list) else [
                loc.strip() for loc in str(excluded_locations).split(",") if loc.strip()
            ]
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
                exclude_locations=exclude_locations_list,
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

    st.dataframe(_safe_df_for_display(processed["employee_df"]), use_container_width=True)


if view_mode == "Employee Lookup":
    st.subheader("Employee Lookup")
    if processed is None:
        st.caption("Run processing to enable lookup.")
    else:
        employee_df = processed["employee_df"].copy()
        employee_df["employee_id"] = employee_df["employee_id"].astype(str)
        employee_df["label"] = (
            employee_df["employee_id"].astype(str)
            + " - "
            + employee_df["firstname"].fillna("").astype(str)
            + " "
            + employee_df["surname"].fillna("").astype(str)
        ).str.strip()
        search_field = st.selectbox("Filter Field", ["Employee Id", "Name", "Location", "Cost Centre"])
        query = st.text_input("Search")
        if query:
            if search_field == "Employee Id":
                filtered = employee_df[employee_df["employee_id"].str.contains(query, case=False, na=False)]
            elif search_field == "Name":
                filtered = employee_df["label"].str.contains(query, case=False, na=False)
                filtered = employee_df[filtered]
            elif search_field == "Location":
                filtered = employee_df["location"].astype(str).str.contains(query, case=False, na=False)
                filtered = employee_df[filtered]
            else:
                filtered = employee_df["cost_centre"].astype(str).str.contains(query, case=False, na=False)
                filtered = employee_df[filtered]
        else:
            filtered = employee_df
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.stop()

st.subheader("4. Manual Adjustments (No Exception)")
exceptions_by_emp: dict[str, list[ExceptionRecord]] = {}
for exc in st.session_state.exceptions:
    exceptions_by_emp.setdefault(str(exc.employee_id), []).append(exc)
if processed is not None:
    # Build absence options + paid/unpaid map for bucket selection
    absence_options = []
    absence_codes = []
    absence_paid_map = {}
    if absence_df is not None and "Short Name" in absence_df.columns:
        temp_codes = (
            absence_df["Short Name"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        temp_desc = (
            absence_df.get("Description", pd.Series([""] * len(temp_codes)))
            .astype(str)
            .str.strip()
            .tolist()
        )
        temp_paid = (
            absence_df.get("Include in Actual Worked Hrs", pd.Series([""]))
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        for code, desc, paid_flag in zip(temp_codes, temp_desc, temp_paid):
            if code:
                absence_codes.append(code)
                absence_options.append(f"{code} - {desc}" if desc else code)
                absence_paid_map[code] = paid_flag == "yes"
    absence_codes = sorted(set(absence_codes))
    st.session_state.absence_options = absence_options
    st.session_state.absence_codes = absence_codes
    st.session_state.absence_paid_map = absence_paid_map

    with st.form("manual_adjustments_form", clear_on_submit=True):
        employee_rows = processed["employee_df"].copy()
        employee_rows["employee_id"] = employee_rows["employee_id"].astype(str)
        inferred_len = int(employee_rows["employee_id"].astype(str).str.len().max() or 0)
        pad_len = int(employee_id_length) if employee_id_length else inferred_len
        employee_rows["label"] = (
            employee_rows["employee_id"].astype(str).apply(
                lambda v: v.zfill(pad_len) if pad_len else v
            )
            + " - "
            + employee_rows["firstname"].fillna("").astype(str)
            + " "
            + employee_rows["surname"].fillna("").astype(str)
        ).str.strip()
        employee_labels = employee_rows["label"].tolist()
        label_to_id = dict(zip(employee_labels, employee_rows["employee_id"].tolist()))
        adj_employee_label = st.selectbox("Employee", employee_labels)
        adj_employee = label_to_id.get(adj_employee_label, "")
        adj_type = st.selectbox("Adjustment Type", ["hours", "money"])
        adj_amount = st.number_input("Adjustment Amount (+/-)", value=0.0, step=0.25)
        adj_bucket = st.selectbox("Absence Bucket", absence_options if absence_options else ["BASIC"])
        adj_comment = st.text_input("Comment (optional)")
        emp_row = None
        if adj_employee:
            matches = employee_rows[employee_rows["employee_id"] == adj_employee]
            if not matches.empty:
                emp_row = matches.iloc[0]
        if emp_row is not None:
            base_totals = _compute_employee_totals(
                emp_row,
                exceptions_by_emp,
                absence_paid_map,
                processed,
            )
            extra = {}
            bucket = adj_bucket.split(" - ")[0]
            if adj_type == "money":
                extra["custom_money_delta"] = adj_amount
                extra["custom_by_code_money"] = {bucket: adj_amount}
            else:
                extra["custom_hours_delta"] = adj_amount
                extra["custom_by_code_hours"] = {bucket: adj_amount}
            after_totals = _compute_employee_totals(
                emp_row,
                exceptions_by_emp,
                absence_paid_map,
                processed,
                extra=extra,
            )
            st.caption(
                f"Pay before adjustment: £{base_totals['total_money']:.2f} → "
                f"after draft: £{after_totals['total_money']:.2f}"
            )
        add_adj = st.form_submit_button("Add Adjustment")
        if add_adj:
            st.session_state.manual_adjustments.append(
                {
                    "employee_id": adj_employee,
                    "type": adj_type,
                    "amount": adj_amount,
                    "bucket": adj_bucket.split(" - ")[0],
                    "comment": adj_comment,
                }
            )
            _save_snapshot(token)
            st.success("Adjustment added")

    if st.session_state.manual_adjustments:
        st.dataframe(pd.DataFrame(st.session_state.manual_adjustments), use_container_width=True, hide_index=True)

    st.subheader("Date Adjustments")
    employee_rows = processed["employee_df"].copy()
    employee_rows["employee_id"] = employee_rows["employee_id"].astype(str)
    inferred_len = int(employee_rows["employee_id"].astype(str).str.len().max() or 0)
    pad_len = int(employee_id_length) if employee_id_length else inferred_len
    employee_rows["label"] = (
        employee_rows["employee_id"].astype(str).apply(
            lambda v: v.zfill(pad_len) if pad_len else v
        )
        + " - "
        + employee_rows["firstname"].fillna("").astype(str)
        + " "
        + employee_rows["surname"].fillna("").astype(str)
    ).str.strip()
    employee_labels = employee_rows["label"].tolist()
    label_to_id = dict(zip(employee_labels, employee_rows["employee_id"].tolist()))

    with st.form("date_adjustments_form", clear_on_submit=True):
        date_adj_employee_label = st.selectbox(
            "Employee (Date Adjustment)",
            employee_labels,
            key="date_adj_employee_label",
        )
        date_adj_employee = label_to_id.get(date_adj_employee_label, "")
        date_adj_date = st.date_input("Adjustment Date")
        date_adj_hours = st.number_input("Adjustment Hours (+/-)", value=0.0, step=0.25)
        date_adj_bucket = st.selectbox(
            "Absence Bucket (Date Adjustment)",
            absence_options if absence_options else ["BASIC"],
        )
        date_adj_comment = st.text_input("Comment (optional, Date Adjustment)")
        add_date_adj = st.form_submit_button("Add Date Adjustment")
        if add_date_adj:
            st.session_state.date_adjustments.append(
                {
                    "employee_id": date_adj_employee,
                    "date": date_adj_date.isoformat() if date_adj_date else "",
                    "hours": date_adj_hours,
                    "bucket": date_adj_bucket.split(" - ")[0],
                    "comment": date_adj_comment,
                }
            )
            _save_snapshot(token)
            st.success("Date adjustment added")

    if st.session_state.date_adjustments:
        st.dataframe(
            pd.DataFrame(st.session_state.date_adjustments),
            use_container_width=True,
            hide_index=True,
        )

    if processed is not None:
        selected_employee_id = label_to_id.get(st.session_state.get("date_adj_employee_label", ""), "")
        if selected_employee_id:
            breakdown = _build_daily_breakdown(
                selected_employee_id,
                processed,
                period_start,
                period_end,
                absence_codes,
                absence_paid_map,
            )
            if not breakdown.empty:
                st.subheader("Daily Breakdown (Selected Employee)")
                st.dataframe(breakdown, use_container_width=True, hide_index=True)
else:
    st.caption("Run processing to enable manual adjustments.")


st.subheader("5. Exceptions")
exceptions: list[ExceptionRecord] = st.session_state.exceptions
open_exceptions = [e for e in exceptions if not e.is_resolved()]

st.write(f"Open exceptions: {len(open_exceptions)}")

absence_options = st.session_state.get("absence_options", [])
absence_codes = st.session_state.get("absence_codes", [])
absence_paid_map = st.session_state.get("absence_paid_map", {})

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
    st.caption("Review the table, then edit one exception at a time using the dropdown below.")
    show_open_only = st.checkbox("Show open exceptions only", value=True)
    exceptions_view = open_exceptions if show_open_only else exceptions

    table_rows = []
    for exc in exceptions_view:
        row = exception_row_by_id.get(exc.exception_id)
        if row:
            emp_id_display = str(row["employee_id"])
            pad_len = int(employee_id_length) if employee_id_length else len(emp_id_display)
            if employee_id_length is None or employee_id_length == 0:
                pad_len = max(pad_len, len(emp_id_display))
            emp_id_display = emp_id_display.zfill(pad_len)
            table_rows.append(
                {
                    "Employee": emp_id_display,
                    "Name": row["name"],
                    "Cost Centre": row["cost_centre"],
                    "Type": row["exception_type"],
                    "Dates / Summary": f"{row['dates']} {row['summary']}".strip(),
                    "Status": row["status"],
                }
            )
    if table_rows:
        table_df = pd.DataFrame(table_rows)
        def _status_color(val: str) -> str:
            if str(val).upper() == "OPEN":
                return "color: #ff4d4f; font-weight: 600;"
            if str(val).upper() == "APPROVED":
                return "color: #52c41a; font-weight: 600;"
            return ""
        styled = table_df.style.applymap(_status_color, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Exception Editors")
    for exc in exceptions_view:
        row = exception_row_by_id.get(exc.exception_id)
        emp_id_display = str(row["employee_id"])
        pad_len = int(employee_id_length) if employee_id_length else len(emp_id_display)
        if employee_id_length is None or employee_id_length == 0:
            pad_len = max(pad_len, len(emp_id_display))
        emp_id_display = emp_id_display.zfill(pad_len)
        label = f"{emp_id_display} - {row['name']} - {row['exception_type']}"
        with st.expander(label, expanded=False):
            actions = _allowed_actions(exc.exception_type)
            selected_action = st.selectbox("Action", actions, key=f"action_{exc.exception_id}")

            extra_fields: dict[str, float] = {}
            if selected_action == "deduct_unpaid_days":
                extra_fields["deduction_days"] = st.number_input(
                    "Deduction days", min_value=0.0, step=0.5, key=f"ded_days_{exc.exception_id}"
                )
                extra_fields["deduction_hours"] = st.number_input(
                    "Deduction hours", min_value=0.0, step=0.25, key=f"ded_hours_{exc.exception_id}"
                )
                bucket = st.selectbox(
                    "Absence Bucket",
                    absence_options if absence_options else ["BASIC"],
                    key=f"ded_bucket_{exc.exception_id}",
                )
                extra_fields["custom_adjustment_bucket"] = bucket.split(" - ")[0]
            elif selected_action == "approve_overtime":
                extra_fields["overtime_hours"] = st.number_input(
                    "Overtime hours", min_value=0.0, step=0.5, key=f"ot_hours_{exc.exception_id}"
                )
            elif selected_action == "custom_adjustment":
                adj_type = st.selectbox(
                    "Adjustment Type", ["hours", "money"], key=f"adj_type_{exc.exception_id}"
                )
                adj_amount = st.number_input(
                    "Adjustment Amount (+/-)", value=0.0, step=0.25, key=f"adj_amount_{exc.exception_id}"
                )
                bucket = st.selectbox(
                    "Absence Bucket",
                    absence_options if absence_options else ["BASIC"],
                    key=f"adj_bucket_{exc.exception_id}",
                )
                extra_fields["custom_adjustment_type"] = adj_type
                extra_fields["custom_adjustment_amount"] = adj_amount
                extra_fields["custom_adjustment_bucket"] = bucket.split(" - ")[0]

            comment = st.text_input("Comment (optional)", key=f"comment_{exc.exception_id}")

            emp = employee_lookup.get(str(exc.employee_id), {})
            weekly_hours = emp.get("weekly_hours") or 0.0
            annual_salary = emp.get("basic_pay_value") or 0.0
            hourly_rate = 0.0
            if emp.get("pay_basis") == "HOURLY":
                hourly_rate = annual_salary
            elif weekly_hours:
                hourly_rate = annual_salary / (weekly_hours * 52.0)
            hours_per_day = (weekly_hours / 5.0) if weekly_hours else 0.0

            preview = ""
            if selected_action == "deduct_unpaid_days":
                d_days = float(extra_fields.get("deduction_days") or 0.0)
                d_hours = float(extra_fields.get("deduction_hours") or 0.0)
                total_hours = d_days * hours_per_day + d_hours
                preview = f"Deduct {total_hours:.2f} hours (~£{total_hours * hourly_rate:.2f})"
            elif selected_action == "custom_adjustment":
                adj_type = extra_fields.get("custom_adjustment_type")
                adj_amount = float(extra_fields.get("custom_adjustment_amount") or 0.0)
                if adj_type == "money":
                    hours_equiv = adj_amount / hourly_rate if hourly_rate else 0.0
                    preview = f"Adjust £{adj_amount:.2f} (≈{hours_equiv:.2f} hours)"
                else:
                    preview = f"Adjust {adj_amount:.2f} hours (≈£{adj_amount * hourly_rate:.2f})"
            elif selected_action == "approve_overtime":
                ot_hours = float(extra_fields.get("overtime_hours") or 0.0)
                preview = f"Overtime {ot_hours:.2f} hours (≈£{ot_hours * hourly_rate:.2f})"
            elif selected_action == "approve_paid":
                preview = "No deduction; pay full salary"
            if preview:
                st.caption(preview)

            if emp is not None:
                base_totals = _compute_employee_totals(
                    emp,
                    exceptions_by_emp,
                    absence_paid_map,
                    processed,
                )
                extra = {}
                if selected_action == "deduct_unpaid_days":
                    extra["deduction_days"] = float(extra_fields.get("deduction_days") or 0.0)
                    extra["deduction_hours"] = float(extra_fields.get("deduction_hours") or 0.0)
                elif selected_action == "approve_overtime":
                    extra["overtime_hours"] = float(extra_fields.get("overtime_hours") or 0.0)
                elif selected_action == "custom_adjustment":
                    adj_type = extra_fields.get("custom_adjustment_type", "hours")
                    adj_amount = float(extra_fields.get("custom_adjustment_amount") or 0.0)
                    bucket = extra_fields.get("custom_adjustment_bucket", "BASIC")
                    if adj_type == "money":
                        extra["custom_money_delta"] = adj_amount
                        extra["custom_by_code_money"] = {bucket: adj_amount}
                    else:
                        extra["custom_hours_delta"] = adj_amount
                        extra["custom_by_code_hours"] = {bucket: adj_amount}
                after_totals = _compute_employee_totals(
                    emp,
                    exceptions_by_emp,
                    absence_paid_map,
                    processed,
                    extra=extra,
                )
                st.caption(
                    f"Pay before adjustment: £{base_totals['total_money']:.2f} → "
                    f"after draft: £{after_totals['total_money']:.2f}"
                )

            if st.button("Save Draft Adjustment", key=f"save_{exc.exception_id}"):
                st.session_state.pending_resolutions[exc.exception_id] = {
                    "action": selected_action,
                    "details": extra_fields,
                    "comment": comment,
                }
                st.success("Draft saved")

            st.subheader("Date Adjustments (This Employee)")
            with st.form(f"date_adjustment_exc_{exc.exception_id}", clear_on_submit=True):
                exc_date = st.date_input("Adjustment Date", key=f"exc_date_{exc.exception_id}")
                exc_hours = st.number_input(
                    "Adjustment Hours (+/-)",
                    value=0.0,
                    step=0.25,
                    key=f"exc_hours_{exc.exception_id}",
                )
                exc_bucket = st.selectbox(
                    "Absence Bucket (Date Adjustment)",
                    absence_options if absence_options else ["BASIC"],
                    key=f"exc_bucket_{exc.exception_id}",
                )
                exc_comment = st.text_input(
                    "Comment (optional, Date Adjustment)",
                    key=f"exc_comment_{exc.exception_id}",
                )
                add_exc_date_adj = st.form_submit_button("Add Date Adjustment")
                if add_exc_date_adj:
                    st.session_state.date_adjustments.append(
                        {
                            "employee_id": str(exc.employee_id),
                            "date": exc_date.isoformat() if exc_date else "",
                            "hours": exc_hours,
                            "bucket": exc_bucket.split(" - ")[0],
                            "comment": exc_comment,
                            "exception_id": exc.exception_id,
                        }
                    )
                    _save_snapshot(token)
                    st.success("Date adjustment added")

            emp_date_adjustments = [
                adj for adj in st.session_state.date_adjustments
                if str(adj.get("employee_id")) == str(exc.employee_id)
            ]
            if emp_date_adjustments:
                st.dataframe(
                    pd.DataFrame(emp_date_adjustments),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Daily Breakdown"):
                breakdown = _build_daily_breakdown(
                    str(exc.employee_id),
                    processed,
                    period_start,
                    period_end,
                    absence_codes,
                    absence_paid_map,
                )
                if breakdown.empty:
                    st.caption("No daily data available.")
                else:
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    if st.button("Apply All Approvals"):
        effective_operator = operator_name.strip() if operator_name.strip() else (app_username or "unknown")
        applied = 0
        for exc in exceptions:
            pending = st.session_state.pending_resolutions.get(exc.exception_id)
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


st.subheader("5. Audit Log")
if st.session_state.audit_log:
    audit_df = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    st.caption("No audit entries yet.")

st.subheader("6. Final Review")
if processed is not None:
    review_rows = []
    for _, row in processed["employee_df"].iterrows():
        employee_id = str(row.get("employee_id"))
        pad_len = int(employee_id_length) if employee_id_length else len(employee_id)
        if employee_id_length is None or employee_id_length == 0:
            pad_len = max(pad_len, len(employee_id))
        employee_id = employee_id.zfill(pad_len)
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
        custom_money_delta = 0.0
        custom_by_code_hours: dict[str, float] = {}
        custom_by_code_money: dict[str, float] = {}
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
                adj_type = resolution.get("custom_adjustment_type", "hours")
                adj_amount = float(resolution.get("custom_adjustment_amount") or 0.0)
                bucket = resolution.get("custom_adjustment_bucket", "BASIC")
                if adj_type == "money":
                    custom_money_delta += adj_amount
                else:
                    custom_hours_delta += adj_amount
                custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + (adj_amount if adj_type == "hours" else 0.0)
                custom_by_code_money[bucket] = custom_by_code_money.get(bucket, 0.0) + (adj_amount if adj_type == "money" else 0.0)

        # Manual adjustments
        for adj in st.session_state.manual_adjustments:
            if str(adj.get("employee_id")) != employee_id:
                continue
            adj_type = adj.get("type", "hours")
            adj_amount = float(adj.get("amount") or 0.0)
            bucket = adj.get("bucket", "BASIC")
            if adj_type == "money":
                custom_money_delta += adj_amount
            else:
                custom_hours_delta += adj_amount
            custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + (adj_amount if adj_type == "hours" else 0.0)
            custom_by_code_money[bucket] = custom_by_code_money.get(bucket, 0.0) + (adj_amount if adj_type == "money" else 0.0)

        # Date adjustments (hours by day)
        for adj in st.session_state.date_adjustments:
            if str(adj.get("employee_id")) != employee_id:
                continue
            adj_amount = float(adj.get("hours") or 0.0)
            bucket = adj.get("bucket", "BASIC")
            custom_hours_delta += adj_amount
            custom_by_code_hours[bucket] = custom_by_code_hours.get(bucket, 0.0) + adj_amount

        hours_per_day = (weekly_hours / 5.0) if weekly_hours else 8.0
        final_base_hours = standard_hours - (deduction_days * hours_per_day) - deduction_hours + custom_hours_delta
        if final_base_hours < 0:
            final_base_hours = 0.0

        # Hourly vs salaried pay rules
        synel_overtime_hours = row.get("overtime_hours_synel") or 0.0
        actual_hours = row.get("actual_hours") or 0.0
        worked_hours = row.get("worked_hours") or 0.0
        regular_hours = max((worked_hours or 0.0), 0.0)

        # Paid/unpaid absence impact for hourly
        abs_map = row.get("absence_days_by_code") or {}
        abs_hours_map = row.get("absence_hours_by_code") or {}
        paid_abs_days = 0.0
        unpaid_abs_days = 0.0
        for code, days in abs_map.items():
            if absence_paid_map.get(code, False):
                paid_abs_days += days
            else:
                unpaid_abs_days += days
        paid_abs_hours = paid_abs_days * hours_per_day
        unpaid_abs_hours = unpaid_abs_days * hours_per_day

        if pay_basis == "SALARIED":
            base_monthly_pay = annual_salary / 12.0
            overtime_money = overtime_hours * hourly_rate
        else:
            # Hourly: pay worked hours + paid absence hours (from BASIC PAY assigned to code)
            paid_abs_hours = sum(
                hours for code, hours in abs_hours_map.items() if absence_paid_map.get(code, False)
            )
            unpaid_abs_hours = sum(
                hours for code, hours in abs_hours_map.items() if not absence_paid_map.get(code, False)
            )
            base_monthly_pay = (regular_hours + paid_abs_hours) * hourly_rate
            overtime_money = synel_overtime_hours * hourly_rate * 1.5

        deduction_money = (deduction_days * hours_per_day + deduction_hours) * hourly_rate
        custom_money = (custom_hours_delta * hourly_rate) + custom_money_delta
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
                "Location": row.get("location", ""),
                "Pay Basis": pay_basis,
                "Annual Salary": round(annual_salary, 2) if pay_basis == "SALARIED" else "",
                "Base Monthly Hours": round(standard_hours, 2),
                "Final Base Hours": round(final_base_hours, 2),
                "Deduction Days": round(deduction_days, 2),
                "Deduction Hours": round(deduction_hours, 2),
                "Regular Hours": "" if pay_basis == "SALARIED" else round(regular_hours, 2),
                "Overtime Hours (Synel)": round(synel_overtime_hours, 2),
                "Overtime Hours (Approved)": round(overtime_hours, 2),
                "Paid Absence Hours": "" if pay_basis == "SALARIED" else round(paid_abs_hours, 2),
                "Unpaid Absence Hours": "" if pay_basis == "SALARIED" else round(unpaid_abs_hours, 2),
                "Custom Hours Delta": round(custom_hours_delta, 2),
                "Custom Money Delta": round(custom_money_delta, 2),
                "Hourly Rate": round(hourly_rate, 4),
                "Base Monthly Pay": round(base_monthly_pay, 2),
                "Deduction Money": round(deduction_money, 2),
                "Custom Money": round(custom_money, 2),
                "Overtime Money": round(overtime_money, 2),
                "Car Allowance": round(car_allowance, 2),
                "FM/FA Weekly": round(fire_marshal_weekly, 2),
                "FM/FA Weeks": round(weeks_in_period, 3),
                "FM/FA Pay": round(fire_marshal_pay, 2),
                "Total Pay": round(total_money, 2),
            }

        # Add absence code breakdown columns
        for code in absence_codes:
            days = abs_map.get(code, 0.0)
            row_out[f"Absence {code} Days"] = round(days, 2)
            # Estimate pay impact per code
            if pay_basis == "SALARIED":
                daily_rate = row.get("daily_rate") or 0.0
                row_out[f"Absence {code} Pay"] = round(days * daily_rate, 2)
            else:
                paid_flag = absence_paid_map.get(code, False)
                hours = abs_hours_map.get(code, 0.0)
                row_out[f"Absence {code} Hours"] = round(hours, 2)
                row_out[f"Absence {code} Pay"] = round(hours * hourly_rate, 2) if paid_flag else 0.0

        # Add custom adjustments by bucket
        for code in absence_codes:
            if custom_by_code_hours.get(code):
                row_out[f"Adj {code} Hours"] = round(custom_by_code_hours.get(code, 0.0), 2)
            if custom_by_code_money.get(code):
                row_out[f"Adj {code} Money"] = round(custom_by_code_money.get(code, 0.0), 2)

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
                ExportConfig(
                    payroll_type=payroll_type,
                    employee_id_length=int(employee_id_length) if employee_id_length else None,
                    excel_safe_ids=True,
                ),
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
