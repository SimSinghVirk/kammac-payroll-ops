from __future__ import annotations

REQUIRED_MAPPING_COLUMNS = [
    "Employee Id",
    "Firstname",
    "Surname",
    "Location",
    "Basic Pay",
    "Hours",
    "COST CENTRE",
    "Car allowance",
    "FM/FA- WEEKLY",
    "CUSTOM GROSS TO NET MAPPING",
    "EXCEPTIONS",
]

REQUIRED_ABSENCE_COLUMNS = [
    "Short Name",
    "Description",
    "Include in Actual Worked Hrs",
]

REQUIRED_PAY_ELEMENTS_COLUMNS = [
    "component_name",
    "rate_code",
    "applies_to_payroll_type",
    "active_flag",
]

SYNEL_REQUIRED_COLUMNS = [
    "Emp No",
    "Employee Name",
    "Date",
    "IN",
    "OUT",
    "IN",
    "OUT",
    "Abs. Half Day",
    "Abs. Half Day",
    "BASIC PAY",
    "TIME & A 1/2",
    "OVERTIME",
    "HOLIDAY PAY",
    "BANK HOLIDAY PAY",
    "SUP - Sickness",
]

SYNEL_ABSENCE_COLS = ["ABS_HALF_DAY_1", "ABS_HALF_DAY_2"]
SYNEL_PUNCH_COLS = ["IN_1", "OUT_1", "IN_2", "OUT_2"]

PAYROLL_TYPES = ["direct_monthly", "admin_monthly"]

EXCEPTION_TYPES = [
    "OVER_HOURS_SALARIED",
    "UNDER_HOURS_SALARIED",
    "NO_PUNCHES_SALARIED",
    "MISSING_PUNCH_DAY",
    "ABSENCE_CODE_UNKNOWN",
    "ABSENCE_CODE_REVIEW",
]

PAID_ABSENCE_CODES_DEFAULT = {"AP", "HP", "SP", "BP", "MAT", "SHP", "PAT"}
UNPAID_ABSENCE_CODES_DEFAULT = {"AUP", "SUP", "DEP", "BUP", "MUP", "SHUP", "UAUP"}

SAGE_EXPORT_COLUMNS = ["Salary ID", "Rate Code", "Hours"]
