from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import re

SCIENTIFIC_NOTATION_RE = re.compile(r"[eE][+-]?\d+")
DIGITS_ONLY_RE = re.compile(r"^\d+$")


@dataclass
class IdCheck:
    value: str | None
    valid: bool
    reason: str | None


def normalize_employee_id(value: Any) -> IdCheck:
    if value is None:
        return IdCheck(None, False, "Employee Id is missing")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return IdCheck(None, False, "Employee Id is missing")
    if not isinstance(value, str):
        return IdCheck(None, False, "Employee Id must be stored as string (no numeric coercion)")

    trimmed = value.strip()
    if trimmed == "":
        return IdCheck(None, False, "Employee Id is blank")
    if SCIENTIFIC_NOTATION_RE.search(trimmed):
        return IdCheck(None, False, "Employee Id appears in scientific notation")
    if not DIGITS_ONLY_RE.match(trimmed):
        return IdCheck(None, False, "Employee Id must contain digits only")

    return IdCheck(trimmed, True, None)


def coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        # Remove common currency symbols and thousand separators.
        cleaned = cleaned.replace("£", "").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text
