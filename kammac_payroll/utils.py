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
        # Handle HH:MM or HH:MM:SS time strings as hours.
        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2]) if len(parts) == 3 else 0
                return hours + minutes / 60.0 + seconds / 3600.0
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


def parse_time_to_hours(value: Any) -> float | None:
    text = safe_str(value)
    if not text:
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return hours + minutes / 60.0 + seconds / 3600.0
