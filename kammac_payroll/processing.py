from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid

import pandas as pd

from .constants import SYNEL_ABSENCE_COLS, SYNEL_PUNCH_COLS
from .exceptions import ExceptionRecord
from .utils import coerce_numeric, normalize_employee_id, safe_str


@dataclass
class ProcessingConfig:
    payroll_type: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    hourly_threshold: float = 100.0
    variance_tolerance: float = 0.5
    ignore_unmapped_zero_activity: bool = False
    employee_id_length: int | None = None


def normalize_synel_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_columns: list[str] = []
    seen: dict[str, int] = {}

    for col in df.columns:
        base = safe_str(col)
        base = base.replace("\n", " ").strip()
        count = seen.get(base, 0) + 1
        seen[base] = count

        if base == "IN":
            cleaned_columns.append(f"IN_{count}")
        elif base == "OUT":
            cleaned_columns.append(f"OUT_{count}")
        elif base == "Abs. Half Day":
            cleaned_columns.append(f"ABS_HALF_DAY_{count}")
        else:
            cleaned_columns.append(base)

    df = df.copy()
    df.columns = cleaned_columns
    return df


def _paid_unpaid_maps(absence_df: pd.DataFrame) -> tuple[set[str], set[str]]:
    paid: set[str] = set()
    unpaid: set[str] = set()
    for _, row in absence_df.iterrows():
        code = safe_str(row.get("Short Name")).upper()
        desc = safe_str(row.get("Description")).lower()
        include = safe_str(row.get("Include in Actual Worked Hrs")).lower()
        if code == "":
            continue
        if "paid" in desc or include == "yes":
            paid.add(code)
        elif "unpaid" in desc or include == "no":
            unpaid.add(code)
    return paid, unpaid


def _extract_absence_days(row: pd.Series, has_second_half: bool) -> dict[str, float]:
    code1 = safe_str(row.get("ABS_HALF_DAY_1")).upper()
    code2 = safe_str(row.get("ABS_HALF_DAY_2")).upper()
    days: dict[str, float] = {}
    if code1:
        days[code1] = days.get(code1, 0.0) + (0.5 if has_second_half else 1.0)
    if code2:
        days[code2] = days.get(code2, 0.0) + 0.5
    return days


def _extract_absence_codes(row: pd.Series) -> list[str]:
    code1 = safe_str(row.get("ABS_HALF_DAY_1")).upper()
    code2 = safe_str(row.get("ABS_HALF_DAY_2")).upper()
    codes = []
    if code1:
        codes.append(code1)
    if code2:
        codes.append(code2)
    return codes


def _row_missing_punch(row: pd.Series) -> bool:
    in1 = safe_str(row.get("IN_1"))
    out1 = safe_str(row.get("OUT_1"))
    in2 = safe_str(row.get("IN_2"))
    out2 = safe_str(row.get("OUT_2"))

    def missing_pair(a: str, b: str) -> bool:
        return (a != "" and b == "") or (a == "" and b != "")

    return missing_pair(in1, out1) or missing_pair(in2, out2)


def _pay_element_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "BASIC PAY",
        "TIME & A 1/2",
        "OVERTIME",
        "HOLIDAY PAY",
        "BANK HOLIDAY PAY",
        "SUP - Sickness",
    ]
    return [col for col in candidates if col in df.columns]


def _overtime_hours(row: pd.Series) -> float:
    # Use TIME & A 1/2 if present, else OVERTIME. Do not double count.
    time_half = coerce_numeric(row.get("TIME & A 1/2")) or 0.0
    overtime = coerce_numeric(row.get("OVERTIME")) or 0.0
    return time_half if time_half > 0 else overtime


def process_run(
    mapping_df: pd.DataFrame,
    absence_df: pd.DataFrame,
    pay_elements_df: pd.DataFrame,
    synel_df: pd.DataFrame,
    config: ProcessingConfig,
    exclude_locations: list[str] | None = None,
) -> dict[str, Any]:
    exceptions: list[ExceptionRecord] = []
    blocking: list[str] = []

    mapping_df = mapping_df.copy()
    synel_df = normalize_synel_columns(synel_df)

    synel_df["Date"] = pd.to_datetime(synel_df["Date"], errors="coerce")

    period_mask = (synel_df["Date"] >= config.period_start) & (
        synel_df["Date"] <= config.period_end
    )
    synel_period = synel_df.loc[period_mask].copy()

    # Ensure employee ids are string-based
    mapping_df["Employee Id"] = mapping_df["Employee Id"].astype(str).str.strip()
    synel_period["Emp No"] = synel_period["Emp No"].astype(str).str.strip()

    # Normalize ID lengths to preserve leading zeros
    id_len = int(config.employee_id_length or 0)
    if not id_len:
        id_lengths = mapping_df["Employee Id"].astype(str).str.len()
        id_len = int(id_lengths.max()) if not id_lengths.empty else 0

    def _pad_id(value: str) -> str:
        if value.isdigit() and id_len:
            return value.zfill(id_len)
        return value

    mapping_df["Employee Id"] = mapping_df["Employee Id"].apply(_pad_id)
    synel_period["Emp No"] = synel_period["Emp No"].apply(_pad_id)

    if "Payroll Type" not in mapping_df.columns:
        if "COST CENTRE" not in mapping_df.columns:
            raise ValueError(
                "Mapping must include either 'Payroll Type' or 'COST CENTRE' to separate direct/admin."
            )
        # Derive payroll type from COST CENTRE values.
        def derive_payroll_type(value: Any) -> str:
            text = safe_str(value).lower()
            if text.startswith("1") or "direct monthly" in text:
                return "direct_monthly"
            if text.startswith("2") or "admin monthly" in text:
                return "admin_monthly"
            if text.startswith("3") or "weekly" in text:
                return "exclude_weekly"
            return "unknown"

        mapping_df = mapping_df.copy()
        mapping_df["Payroll Type"] = mapping_df["COST CENTRE"].apply(derive_payroll_type)

    if (mapping_df["Payroll Type"] == "unknown").any():
        blocking.append("Unable to derive payroll type from COST CENTRE for some rows.")

    mapping_filtered = mapping_df[mapping_df["Payroll Type"] == config.payroll_type].copy()
    if exclude_locations:
        mapping_filtered = mapping_filtered[
            ~mapping_filtered["Location"].astype(str).str.strip().isin(exclude_locations)
        ]

    # Join Synel rows to mapping
    joined = synel_period.merge(
        mapping_filtered,
        how="left",
        left_on="Emp No",
        right_on="Employee Id",
        suffixes=("", "_map"),
    )

    # Track unmapped
    unmapped = joined[joined["Employee Id"].isna()]["Emp No"].unique().tolist()
    if unmapped:
        # Exclude employees that are mapped to other payroll types (e.g., weekly).
        all_mapping_ids = set(mapping_df["Employee Id"].astype(str).str.strip())
        unmapped_true = [emp for emp in unmapped if str(emp) not in all_mapping_ids]

        if unmapped_true and config.ignore_unmapped_zero_activity:
            pay_element_cols = _pay_element_columns(synel_period)

            def _has_activity(emp: str) -> bool:
                rows = synel_period[synel_period["Emp No"].astype(str).str.strip() == str(emp)]
                if rows.empty:
                    return False
                hours = rows[pay_element_cols].apply(
                    lambda r: sum((coerce_numeric(v) or 0.0) for v in r), axis=1
                ).sum()
                if hours > 0:
                    return True
                punch_cols_present = [c for c in SYNEL_PUNCH_COLS if c in rows.columns]
                punches_present = False
                if punch_cols_present:
                    punches_present = (
                        rows[punch_cols_present]
                        .fillna("")
                        .astype(str)
                        .apply(lambda r: any(v.strip() != "" for v in r), axis=1)
                        .any()
                    )
                if punches_present:
                    return True
                abs_cols_present = [c for c in SYNEL_ABSENCE_COLS if c in rows.columns]
                abs_present = False
                if abs_cols_present:
                    abs_present = (
                        rows[abs_cols_present]
                        .fillna("")
                        .astype(str)
                        .apply(lambda r: any(v.strip() != "" for v in r), axis=1)
                        .any()
                    )
                return bool(abs_present)

            unmapped_active = [emp for emp in unmapped_true if _has_activity(emp)]
            if unmapped_active:
                blocking.append(
                    "Unmapped Synel employees in-period with activity: "
                    + ", ".join([str(x) for x in unmapped_active[:10]])
                )
        elif unmapped_true:
            blocking.append(
                "Unmapped Synel employees in-period: "
                + ", ".join([str(x) for x in unmapped_true[:10]])
            )

    # Pay basis classification
    def classify_pay_basis(row: pd.Series) -> dict[str, Any]:
        basic_pay = coerce_numeric(row.get("Basic Pay"))
        if basic_pay is None:
            return {
                "pay_basis": "INVALID",
                "basic_pay_value": None,
                "rule": "missing_or_non_numeric",
            }
        if basic_pay <= config.hourly_threshold:
            return {
                "pay_basis": "HOURLY",
                "basic_pay_value": basic_pay,
                "rule": f"<= {config.hourly_threshold}",
            }
        return {
            "pay_basis": "SALARIED",
            "basic_pay_value": basic_pay,
            "rule": f"> {config.hourly_threshold}",
        }

    mapping_filtered["_pay_basis_meta"] = mapping_filtered.apply(classify_pay_basis, axis=1)
    mapping_filtered["pay_basis"] = mapping_filtered["_pay_basis_meta"].apply(
        lambda x: x["pay_basis"]
    )
    mapping_filtered["basic_pay_value"] = mapping_filtered["_pay_basis_meta"].apply(
        lambda x: x["basic_pay_value"]
    )
    mapping_filtered["pay_basis_rule"] = mapping_filtered["_pay_basis_meta"].apply(
        lambda x: x["rule"]
    )

    # Absence mappings
    paid_codes, unpaid_codes = _paid_unpaid_maps(absence_df)
    absence_lookup = {
        safe_str(row.get("Short Name")).upper(): row
        for _, row in absence_df.iterrows()
        if safe_str(row.get("Short Name")) != ""
    }

    # Aggregate per employee
    pay_element_cols = _pay_element_columns(synel_period)
    synel_period["_row_total_hours"] = synel_period[pay_element_cols].apply(
        lambda r: sum((coerce_numeric(v) or 0.0) for v in r), axis=1
    )
    if "BASIC PAY" in synel_period.columns:
        synel_period["_row_basic_hours"] = synel_period["BASIC PAY"].apply(lambda v: coerce_numeric(v) or 0.0)
    else:
        synel_period["_row_basic_hours"] = 0.0
    synel_period["_row_overtime_hours"] = synel_period.apply(_overtime_hours, axis=1)

    employee_groups = synel_period.groupby("Emp No", dropna=False)

    employee_summary: list[dict[str, Any]] = []
    exception_id_counter = 0

    for emp_id, group in employee_groups:
        emp_id_check = normalize_employee_id(emp_id)
        if not emp_id_check.valid:
            continue
        emp_id = emp_id_check.value or emp_id

        mapping_row = mapping_filtered[mapping_filtered["Employee Id"] == emp_id]
        if mapping_row.empty:
            continue
        mapping_row = mapping_row.iloc[0]

        pay_basis = mapping_row["pay_basis"]
        weekly_hours = coerce_numeric(mapping_row.get("Hours"))

        standard_monthly_hours = None
        if weekly_hours is not None:
            standard_monthly_hours = round(weekly_hours * 52.0 / 12.0, 2)
        if pay_basis == "INVALID":
            blocking.append(f"Invalid Basic Pay for employee {emp_id}")
        if pay_basis == "SALARIED" and (weekly_hours is None or weekly_hours <= 0):
            blocking.append(f"Missing/invalid weekly contract hours for {emp_id}")

        actual_hours = group["_row_total_hours"].sum()
        overtime_hours = group["_row_overtime_hours"].sum()

        # Absence days per employee
        absence_days: dict[str, float] = {}
        unpaid_absence_days = 0.0
        paid_absence_days = 0.0
        unknown_absence_rows: list[dict[str, Any]] = []
        absence_hours_by_code: dict[str, float] = {}
        worked_hours = 0.0

        has_second_abs = "ABS_HALF_DAY_2" in group.columns
        for _, row in group.iterrows():
            days = _extract_absence_days(row, has_second_abs)
            if not days:
                # If no absence code, treat BASIC PAY hours as worked hours (hourly)
                worked_hours += coerce_numeric(row.get("_row_basic_hours")) or 0.0
                continue
            for code, day_count in days.items():
                absence_days[code] = absence_days.get(code, 0.0) + day_count
                if code in paid_codes:
                    paid_absence_days += day_count
                elif code in unpaid_codes:
                    unpaid_absence_days += day_count
                else:
                    unknown_absence_rows.append(
                        {
                            "date": row.get("Date"),
                            "code": code,
                        }
                    )
            # Allocate BASIC PAY hours to absence bucket(s) to avoid double count
            codes = _extract_absence_codes(row)
            basic_hours = coerce_numeric(row.get("_row_basic_hours")) or 0.0
            if codes and basic_hours > 0:
                per_code = basic_hours / len(codes)
                for code in codes:
                    absence_hours_by_code[code] = absence_hours_by_code.get(code, 0.0) + per_code
            elif basic_hours > 0:
                worked_hours += basic_hours

        # Missing punch detection
        missing_punch_dates: list[pd.Timestamp] = []
        for _, row in group.iterrows():
            if not _row_missing_punch(row):
                continue
            # If unpaid absence explains missing punch, skip creating missing-punch exception
            day_absences = _extract_absence_days(row, has_second_abs)
            unpaid_present = any(code in unpaid_codes for code in day_absences.keys())
            if unpaid_present:
                continue
            missing_punch_dates.append(row.get("Date"))

        # Exception creation
        def add_exception(exception_type: str, details: dict[str, Any]) -> None:
            nonlocal exception_id_counter
            exception_id_counter += 1
            exceptions.append(
                ExceptionRecord(
                    exception_id=str(uuid.uuid4()),
                    employee_id=emp_id,
                    exception_type=exception_type,
                    details=details,
                    created_at=datetime.utcnow(),
                )
            )

        if unknown_absence_rows:
            add_exception(
                "ABSENCE_CODE_UNKNOWN",
                {"rows": unknown_absence_rows},
            )

        if pay_basis == "SALARIED" and config.payroll_type == "admin_monthly":
            for date_value in missing_punch_dates:
                add_exception(
                    "MISSING_PUNCH_DAY",
                    {"date": date_value},
                )

        # No punches salaried
        if pay_basis == "SALARIED":
            if group.empty:
                add_exception("NO_PUNCHES_SALARIED", {"reason": "no rows in period"})
            else:
                punch_cols_present = [c for c in SYNEL_PUNCH_COLS if c in group.columns]
                if punch_cols_present:
                    punches_empty = (
                        group[punch_cols_present]
                        .fillna("")
                        .astype(str)
                        .apply(lambda r: all(v.strip() == "" for v in r), axis=1)
                        .all()
                    )
                else:
                    punches_empty = False
                hours_zero = group["_row_total_hours"].sum() == 0
                if punches_empty and hours_zero:
                    add_exception(
                        "NO_PUNCHES_SALARIED",
                        {"reason": "no punches and zero hours"},
                    )

        # Over/under hours exceptions
        if pay_basis == "SALARIED" and standard_monthly_hours is not None and config.payroll_type == "admin_monthly":
            if actual_hours < standard_monthly_hours - config.variance_tolerance:
                add_exception(
                    "UNDER_HOURS_SALARIED",
                    {
                        "actual_hours": actual_hours,
                        "expected_hours": standard_monthly_hours,
                        "tolerance": config.variance_tolerance,
                    },
                )

        exception_id_counter += 1

        # Daily rate divisor
        exceptions_flag = safe_str(mapping_row.get("EXCEPTIONS")).strip().lower()
        divisor = 182.5 if exceptions_flag == "4x4" else 260.0
        annual_salary = mapping_row.get("basic_pay_value")
        daily_rate = None
        if annual_salary is not None:
            daily_rate = annual_salary / divisor

        employee_summary.append(
            {
                "employee_id": emp_id,
                "firstname": mapping_row.get("Firstname"),
                "surname": mapping_row.get("Surname"),
                "location": mapping_row.get("Location"),
                "cost_centre": mapping_row.get("COST CENTRE"),
                "basic_pay_value": mapping_row.get("basic_pay_value"),
                "car_allowance": coerce_numeric(mapping_row.get("Car allowance")) or 0.0,
                "fm_fa_weekly": coerce_numeric(mapping_row.get("FM/FA- WEEKLY")) or 0.0,
                "pay_basis": pay_basis,
                "weekly_hours": weekly_hours,
                "standard_monthly_hours": standard_monthly_hours,
                "actual_hours": actual_hours,
                "overtime_hours_synel": overtime_hours,
                "paid_absence_days": paid_absence_days,
                "unpaid_absence_days": unpaid_absence_days,
                "absence_days_by_code": absence_days,
                "absence_hours_by_code": absence_hours_by_code,
                "worked_hours": worked_hours,
                "daily_rate_divisor": divisor,
                "daily_rate": daily_rate,
            }
        )

    employee_df = pd.DataFrame(employee_summary)

    return {
        "synel_period": synel_period,
        "joined": joined,
        "employee_df": employee_df,
        "exceptions": exceptions,
        "unmapped_employees": unmapped,
        "blocking": blocking,
    }
