from __future__ import annotations

from typing import Any
import io
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
except Exception:  # pragma: no cover - optional at runtime
    gspread = None
    Credentials = None
    build = None
    MediaIoBaseUpload = None


GOOGLE_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.file",
]


def _service_account_from_info(info: dict[str, Any]):
    if Credentials is None:
        raise RuntimeError("Google credentials not available")
    return Credentials.from_service_account_info(info, scopes=GOOGLE_SHEETS_SCOPES)


def _dedupe_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for header in headers:
        name = str(header).strip()
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count == 1:
            deduped.append(name)
        else:
            deduped.append(f"{name}_{count}")
    return deduped


def load_sheet_as_df(spreadsheet_id: str, tab_name: str, credentials_info: dict[str, Any]) -> pd.DataFrame:
    if gspread is None:
        raise RuntimeError("gspread not installed")
    creds = _service_account_from_info(credentials_info)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(spreadsheet_id).worksheet(tab_name)

    # Use raw values to tolerate duplicate headers, then dedupe them.
    values = sheet.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = _dedupe_headers(values[0])
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


def upload_bytes_to_drive(
    folder_id: str,
    filename: str,
    content_bytes: bytes,
    mime_type: str,
    credentials_info: dict[str, Any],
) -> str:
    if build is None:
        raise RuntimeError("googleapiclient not installed")
    creds = _service_account_from_info(credentials_info)
    service = build("drive", "v3", credentials=creds)
    media = MediaIoBaseUpload(io.BytesIO(content_bytes), mimetype=mime_type)
    file_metadata = {"name": filename, "parents": [folder_id]}
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return str(file.get("id"))
