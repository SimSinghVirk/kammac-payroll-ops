from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from .constants import (
    REQUIRED_MAPPING_COLUMNS,
    REQUIRED_ABSENCE_COLUMNS,
    REQUIRED_PAY_ELEMENTS_COLUMNS,
    SYNEL_REQUIRED_COLUMNS,
)
from .utils import normalize_employee_id


@dataclass
class ValidationResult:
    blocking: list[str]
    warnings: list[str]

    def is_ok(self) -> bool:
        return len(self.blocking) == 0


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [col for col in required if col not in df.columns]
    return missing


def validate_mapping(mapping_df: pd.DataFrame) -> ValidationResult:
    blocking: list[str] = []
    warnings: list[str] = []

    missing = _missing_columns(mapping_df, REQUIRED_MAPPING_COLUMNS)
    if missing:
        blocking.append(f"Mapping missing required columns: {', '.join(missing)}")
        return ValidationResult(blocking, warnings)

    if "Payroll Type" not in mapping_df.columns and "COST CENTRE" not in mapping_df.columns:
        blocking.append("Mapping must include 'Payroll Type' or 'COST CENTRE'")

    if mapping_df["Employee Id"].duplicated().any():
        dupes = mapping_df[mapping_df["Employee Id"].duplicated()]["Employee Id"].astype(str).tolist()
        blocking.append(f"Duplicate Employee Id in mapping: {', '.join(dupes[:10])}")

    invalid_ids = []
    for raw in mapping_df["Employee Id"].tolist():
        check = normalize_employee_id(raw)
        if not check.valid:
            invalid_ids.append(str(raw))
    if invalid_ids:
        blocking.append(
            "Invalid Employee Id values in mapping (must be string digits, no scientific notation)."
        )

    return ValidationResult(blocking, warnings)


def validate_absences(absence_df: pd.DataFrame) -> ValidationResult:
    blocking: list[str] = []
    warnings: list[str] = []
    missing = _missing_columns(absence_df, REQUIRED_ABSENCE_COLUMNS)
    if missing:
        blocking.append(f"Absence mapping missing required columns: {', '.join(missing)}")
    return ValidationResult(blocking, warnings)


def validate_pay_elements(pay_elements_df: pd.DataFrame) -> ValidationResult:
    blocking: list[str] = []
    warnings: list[str] = []
    missing = _missing_columns(pay_elements_df, REQUIRED_PAY_ELEMENTS_COLUMNS)
    if missing:
        blocking.append(f"Pay elements missing required columns: {', '.join(missing)}")
    return ValidationResult(blocking, warnings)


def validate_synel(synel_df: pd.DataFrame) -> ValidationResult:
    blocking: list[str] = []
    warnings: list[str] = []

    # We validate after column normalization. Ensure core columns exist.
    missing = _missing_columns(synel_df, [
        "Emp No",
        "Date",
        "IN_1",
        "OUT_1",
        "ABS_HALF_DAY_1",
    ])
    if missing:
        blocking.append(f"Synel file missing required columns after normalization: {', '.join(missing)}")
        return ValidationResult(blocking, warnings)

    optional = [
        "IN_2",
        "OUT_2",
        "ABS_HALF_DAY_2",
    ]
    optional_missing = [col for col in optional if col not in synel_df.columns]
    if optional_missing:
        warnings.append(f"Synel optional columns missing (ok): {', '.join(optional_missing)}")

    invalid_ids = []
    for raw in synel_df["Emp No"].tolist():
        check = normalize_employee_id(raw)
        if not check.valid:
            invalid_ids.append(str(raw))
    if invalid_ids:
        preview = ", ".join(invalid_ids[:10])
        blocking.append(
            "Invalid Emp No values in Synel (must be string digits, no scientific notation). "
            f"Examples: {preview}. "
            "Export Emp No as text to preserve leading zeros."
        )

    return ValidationResult(blocking, warnings)
