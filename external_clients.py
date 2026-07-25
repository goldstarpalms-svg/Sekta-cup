from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import requests


class ExternalAPIError(RuntimeError):
    """Raised when an external provider returns an error or credentials are missing."""


@dataclass
class APIResponse:
    data: Any
    status_code: int
    headers: dict[str, str]


class PinnacleClient:
    """Minimal Pinnacle API client scaffold.

    Requires approved Pinnacle API credentials. Keep credentials in environment
    variables or Streamlit secrets; do not commit them to Git.

    Common environment variable names used by this project:
    - PINNACLE_USERNAME
    - PINNACLE_PASSWORD
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        base_url: str = "https://api.pinnacle.com",
    ) -> None:
        self.username = username or os.getenv("PINNACLE_USERNAME")
        self.password = password or os.getenv("PINNACLE_PASSWORD")
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.username or not self.password:
            raise ExternalAPIError(
                "Missing Pinnacle credentials. Set PINNACLE_USERNAME and PINNACLE_PASSWORD."
            )
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    def request(self, path: str, params: dict[str, Any] | None = None) -> APIResponse:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.get(url, headers=self._headers(), params=params or {}, timeout=30)
        if not response.ok:
            raise ExternalAPIError(f"Pinnacle API error {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
        except Exception:
            data = response.text
        return APIResponse(data=data, status_code=response.status_code, headers=dict(response.headers))


class BetfairClient:
    """Minimal Betfair API-NG JSON-RPC client scaffold.

    Requires a Betfair app key and session token. Keep credentials in environment
    variables or Streamlit secrets; do not commit them to Git.

    Common environment variable names used by this project:
    - BETFAIR_APP_KEY
    - BETFAIR_SESSION_TOKEN
    """

    def __init__(
        self,
        app_key: str | None = None,
        session_token: str | None = None,
        rpc_url: str = "https://api.betfair.com/exchange/betting/json-rpc/v1",
    ) -> None:
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY")
        self.session_token = session_token or os.getenv("BETFAIR_SESSION_TOKEN")
        self.rpc_url = rpc_url

    def _headers(self) -> dict[str, str]:
        if not self.app_key or not self.session_token:
            raise ExternalAPIError(
                "Missing Betfair credentials. Set BETFAIR_APP_KEY and BETFAIR_SESSION_TOKEN."
            )
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def call(self, method: str, params: dict[str, Any] | None = None) -> APIResponse:
        payload = {
            "jsonrpc": "2.0",
            "method": f"SportsAPING/v1.0/{method}",
            "params": params or {},
            "id": 1,
        }
        response = requests.post(self.rpc_url, headers=self._headers(), json=payload, timeout=30)
        if not response.ok:
            raise ExternalAPIError(f"Betfair API error {response.status_code}: {response.text[:500]}")
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise ExternalAPIError(f"Betfair API error: {data['error']}")
        return APIResponse(data=data.get("result", data), status_code=response.status_code, headers=dict(response.headers))
