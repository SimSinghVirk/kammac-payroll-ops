from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from .constants import SAGE_EXPORT_COLUMNS


@dataclass
class ExportConfig:
    payroll_type: str
    employee_id_length: int | None = None
    excel_safe_ids: bool = True


def _format_employee_id(value: Any, pad_len: int | None, excel_safe: bool) -> str:
    text = "" if value is None else str(value).strip()
    if pad_len and text.isdigit():
        text = text.zfill(pad_len)
    if excel_safe:
        return f'="{text}"' if text else ""
    return text


def _pick_rate_code(pay_elements_df: pd.DataFrame, component_name: str, payroll_type: str) -> str:
    df = pay_elements_df.copy()
    df = df[df["component_name"] == component_name]
    df = df[df["active_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"]) ]
    if payroll_type != "both":
        df = df[df["applies_to_payroll_type"].isin([payroll_type, "both"])]
    if df.empty:
        raise ValueError(f"No active rate code found for component {component_name}")
    return str(df.iloc[0]["rate_code"]).strip()


def build_sage_export(
    employee_df: pd.DataFrame,
    exceptions: list[Any],
    pay_elements_df: pd.DataFrame,
    config: ExportConfig,
) -> pd.DataFrame:
    export_rows: list[dict[str, Any]] = []

    if config.payroll_type == "direct_monthly":
        base_component = "BASE_SALARY_DIRECT"
    else:
        base_component = "BASE_SALARY_ADMIN"

    base_rate_code = _pick_rate_code(pay_elements_df, base_component, config.payroll_type)
    overtime_rate_code = None
    try:
        overtime_rate_code = _pick_rate_code(pay_elements_df, "SALARIED_OVERTIME", config.payroll_type)
    except ValueError:
        overtime_rate_code = None

    exceptions_by_emp: dict[str, list[Any]] = {}
    for exc in exceptions:
        exceptions_by_emp.setdefault(exc.employee_id, []).append(exc)

    for _, row in employee_df.iterrows():
        if row.get("pay_basis") != "SALARIED":
            continue

        employee_id = _format_employee_id(
            row.get("employee_id"),
            config.employee_id_length,
            config.excel_safe_ids,
        )
        standard_hours = row.get("standard_monthly_hours")
        weekly_hours = row.get("weekly_hours")

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

        final_base_hours = standard_hours
        if weekly_hours is not None:
            hours_per_day = float(weekly_hours) / 5.0
            final_base_hours = standard_hours - deduction_days * hours_per_day - deduction_hours + custom_hours_delta

        if final_base_hours < 0:
            raise ValueError(f"Final base hours < 0 for employee {employee_id}")

        export_rows.append(
            {
                "Salary ID": employee_id,
                "Rate Code": base_rate_code,
                "Hours": round(final_base_hours, 2) if final_base_hours is not None else 0,
                "Location": row.get("location"),
            }
        )

        if overtime_hours > 0:
            rate_code = overtime_rate_code or base_rate_code
            export_rows.append(
                {
                    "Salary ID": employee_id,
                    "Rate Code": rate_code,
                    "Hours": round(overtime_hours, 2),
                    "Location": row.get("location"),
                }
            )

    return pd.DataFrame(export_rows, columns=SAGE_EXPORT_COLUMNS)
