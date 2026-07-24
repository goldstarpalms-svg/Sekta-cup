from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

OFFICIAL_SETKA_URL = "https://tabletennis.setkacup.com/en/"


@dataclass
class SetkaSiteStatus:
    ok: bool
    status_code: int | None
    final_url: str
    title: str | None
    error: str | None = None


def fetch_official_site_status(url: str = OFFICIAL_SETKA_URL) -> SetkaSiteStatus:
    """Lightweight status check for the official Setka Cup website.

    The official site may be rendered dynamically and may not expose a public
    data API. This function intentionally checks availability only. If you have
    permission or an official feed/API, plug it into this module.
    """
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "SetkaPredictionApp/1.0 (+https://github.com/)"
            },
        )
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.I | re.S)
        title = None
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        return SetkaSiteStatus(
            ok=response.ok,
            status_code=response.status_code,
            final_url=response.url,
            title=title,
        )
    except Exception as exc:
        return SetkaSiteStatus(
            ok=False,
            status_code=None,
            final_url=url,
            title=None,
            error=str(exc),
        )


def status_as_dict(status: SetkaSiteStatus) -> dict[str, Any]:
    return {
        "ok": status.ok,
        "status_code": status.status_code,
        "final_url": status.final_url,
        "title": status.title,
        "error": status.error,
    }
