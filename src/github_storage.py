from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

GITHUB_API = "https://api.github.com"


def github_storage_enabled() -> bool:
    return bool(os.getenv("GITHUB_STORAGE_TOKEN") and os.getenv("GITHUB_STORAGE_REPO"))


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_STORAGE_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "SetkaPredictionApp/1.0",
    }


def _repo() -> str:
    return os.getenv("GITHUB_STORAGE_REPO", "goldstarpalms-svg/Sekta-cup")


def _branch() -> str:
    return os.getenv("GITHUB_STORAGE_BRANCH", "main")


def _storage_prefix() -> str:
    return os.getenv("GITHUB_STORAGE_PREFIX", "data/app_state")


def github_path(file_name: str | Path) -> str:
    name = Path(file_name).name
    return f"{_storage_prefix().strip('/')}/{name}"


def get_file(path: str) -> tuple[Optional[str], Optional[str]]:
    """Return decoded file content and sha, or (None, None) if missing."""
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    response = requests.get(url, headers=_headers(), params={"ref": _branch()}, timeout=20)
    if response.status_code == 404:
        return None, None
    response.raise_for_status()
    data = response.json()
    content = base64.b64decode(data.get("content", "")).decode("utf-8")
    return content, data.get("sha")


def put_file(path: str, content: str, message: str) -> None:
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    _, sha = get_file(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": _branch(),
    }
    if sha:
        payload["sha"] = sha
    response = requests.put(url, headers=_headers(), json=payload, timeout=30)
    response.raise_for_status()


def load_csv(file_name: str | Path) -> pd.DataFrame:
    if not github_storage_enabled():
        return pd.DataFrame()
    content, _ = get_file(github_path(file_name))
    if not content:
        return pd.DataFrame()
    from io import StringIO

    return pd.read_csv(StringIO(content))


def save_csv(frame: pd.DataFrame, file_name: str | Path, message: str) -> None:
    if not github_storage_enabled():
        return
    put_file(github_path(file_name), frame.to_csv(index=False), message)


def delete_file(file_name: str | Path, message: str) -> None:
    if not github_storage_enabled():
        return
    path = github_path(file_name)
    _, sha = get_file(path)
    if not sha:
        return
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    payload = {"message": message, "sha": sha, "branch": _branch()}
    response = requests.delete(url, headers=_headers(), json=payload, timeout=30)
    response.raise_for_status()
