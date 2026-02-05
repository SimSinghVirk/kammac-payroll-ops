from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExceptionRecord:
    exception_id: str
    employee_id: str
    exception_type: str
    details: dict[str, Any]
    created_at: datetime
    status: str = "OPEN"
    resolution: dict[str, Any] | None = None
    prepared_by: str | None = None
    prepared_at: datetime | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    comments: list[dict[str, Any]] = field(default_factory=list)

    def is_resolved(self) -> bool:
        return self.status == "APPROVED"
