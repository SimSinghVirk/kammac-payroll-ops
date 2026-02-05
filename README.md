# Kammac Monthly Payroll Ops (Streamlit)

This is a Streamlit app to run monthly payroll processing for Kammac Payroll, following the provided requirements spec.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Required Google Sheets Tabs

By default, the app expects these tabs in a single Google Sheet:

- `AI_PAY_MAPPING_SIM` (mapping)
- `ABSENCES_KAMMAC` (absence code mapping)
- `PAY_ELEMENTS_KAMMAC` (rate code mapping)

If you keep each mapping in a separate Google Sheet, use the optional Sheet ID overrides in the sidebar.

### Mapping tab required columns

The mapping tab **must** include a `Payroll Type` column with values:

- `direct_monthly`
- `admin_monthly`

This is required to prevent mixing the two payroll types.

## Google Service Account

Provide credentials via Streamlit secrets:

```
GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

Or paste the JSON in the app’s expander.

## Export

The app produces a Sage import CSV with columns:

- `Salary ID`
- `Rate Code`
- `Hours`

It can upload the export to a Google Drive folder if a folder ID is supplied.
