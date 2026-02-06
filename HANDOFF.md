# Kammac Payroll Ops — Handoff

## Project
- Repo: https://github.com/SimSinghVirk/kammac-payroll-ops
- Deployed app: https://kammac-payroll-ops-yv3sdhf66pkfuqf5ghbrze.streamlit.app/
- Streamlit secrets:
  - `APP_USERNAME = "simsingh"`
  - `APP_PASSWORD = "kammac1997"`
  - `GOOGLE_SERVICE_ACCOUNT_JSON = "..."`

## Data Sources
- **Mapping sheet** tab: `AI_PAY_MAPPING_SIM`
- **Absences** tab: `Absences`
- **Pay elements** tab: `elements`
- **Synel monthly export** with columns:
  - Emp No, Date, IN/OUT, Abs. Half Day, BASIC PAY, TIME & A 1/2, OVERTIME, HOLIDAY PAY, BANK HOLIDAY PAY, SUP - Sickness

## Payroll Type
Derived from `COST CENTRE`:
- 1 → `direct_monthly`
- 2 → `admin_monthly`
- 3 → weekly (excluded)

## Employee IDs
- Always treated as strings.
- UI pads with zeros using **Employee ID length** (default 5).

## Core Logic
### Salaried
- Base pay fixed: **Annual Salary / 12**.
- Under‑hours exceptions for **all salaried** (admin + direct).
- Missing punch exceptions **admin only**.
- Overtime paid only if approved.

### Hourly
- Regular hours from BASIC PAY; if empty and no absence code, compute from IN/OUT punches.
- Overtime from Synel: **TIME & A 1/2** if present else **OVERTIME**.
- Overtime pay = hours × 1.5 × hourly rate.

### Absence buckets
- Absence codes from `Absences` sheet.
- Paid/unpaid uses `Include in Actual Worked Hrs` (Yes/No).
- BASIC PAY hours allocated to absence bucket if code exists (no double‑count).

## Final Review
Includes:
- Annual Salary, Base Monthly Hours, Final Base Hours
- Regular/Ot/Absence hours
- Absence {code} Days/Hours/Pay (paid only)
- Car allowance (monthly) and FM/FA weekly (whole weeks)
- Custom adjustments (hours or money)

## UI Workflow
- Manual Adjustments (No Exception) above Exceptions
- Exceptions table + per‑employee editor
- Apply All Approvals commits changes
- Employee Lookup view exists (parked)

## Known Gaps
- Sage export still only base + overtime; absence buckets not exported (needs rate codes)
- UI styling still functional, not polished

## Recent Changes (2026-02-06)
- Added Location to Sage export and Final Review table.
- Prevented punch fallback from double-counting overtime-only rows.
- Exported Excel-safe, padded Employee IDs (e.g., `=\"00012\"`) for CSVs.
- Added pay preview (before/after draft) in manual adjustments and exception editors.
- Added per-date adjustments and daily breakdown (all dates, includes 0s).
- Removed debug panels (credentials/diagnostics) for cleaner UI.
- Fixed nested expander error in exception editors.
- Auto-rerun after sheet auto-load so Exclude Locations appears immediately.
- Added pay type selector (regular vs overtime) for adjustments and date adjustments.
- Added OTHER pay bucket for adjustments.
