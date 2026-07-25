from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests

OFFICIAL_SETKA_URL = "https://tabletennis.setkacup.com/en/"
SETKA_API_BASE = "https://tabletennis.setkacup.com/api"


@dataclass
class SetkaSiteStatus:
    ok: bool
    status_code: int | None
    final_url: str
    title: str | None
    error: str | None = None


def fetch_official_site_status(url: str = OFFICIAL_SETKA_URL) -> SetkaSiteStatus:
    """Lightweight status check for the official Setka Cup website."""
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "SetkaPredictionApp/1.0 (+https://github.com/)"},
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


def _get_json(path: str, params: dict[str, Any] | None = None, timeout: int = 20) -> Any:
    url = path if path.startswith("http") else f"{SETKA_API_BASE}{path}"
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "SetkaPredictionApp/1.0 (+https://github.com/)"},
    )
    response.raise_for_status()
    return response.json()


def player_name(player: dict[str, Any] | None) -> str:
    if not player:
        return ""
    return f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()


def status_label(status_id: int | str | None) -> str:
    mapping = {
        1: "Scheduled",
        2: "Live",
        3: "Finished",
        4: "Cancelled",
        5: "Technical",
    }
    try:
        key = int(status_id) if status_id is not None else None
    except Exception:
        key = None
    return mapping.get(key, f"Status {status_id}" if status_id is not None else "Unknown")


def fetch_locations(locale: str = "en") -> pd.DataFrame:
    """Fetch official Setka location/hall metadata."""
    data = _get_json(f"/Locations/{locale}")
    rows = []
    for item in data or []:
        rows.append(
            {
                "location_id": item.get("id") or item.get("locationId"),
                "location": item.get("name") or item.get("locationName") or item.get("title"),
                "official": item.get("official"),
                "color": item.get("color"),
            }
        )
    return pd.DataFrame(rows)


def location_map(locale: str = "en") -> dict[int, str]:
    df = fetch_locations(locale)
    if df.empty:
        return {}
    return {
        int(row.location_id): str(row.location)
        for row in df.itertuples(index=False)
        if pd.notna(row.location_id)
    }


def fetch_nearest_matches(locale: str = "en") -> pd.DataFrame:
    """Fetch official upcoming/nearest Setka matches."""
    data = _get_json(f"/Matches/nearest/{locale}")
    rows = []
    for item in data or []:
        rows.append(
            {
                "match_id": item.get("matchId"),
                "location_id": item.get("locationId"),
                "start_date_utc": item.get("startDate"),
                "player1_id": item.get("player1Id"),
                "player1": f"{item.get('player1FirstName', '')} {item.get('player1LastName', '')}".strip(),
                "player2_id": item.get("player2Id"),
                "player2": f"{item.get('player2FirstName', '')} {item.get('player2LastName', '')}".strip(),
                "status_id": 1,
                "status": "Scheduled",
            }
        )
    return pd.DataFrame(rows)


def fetch_live_matches(locale: str = "en") -> pd.DataFrame:
    """Fetch official live widget matches from Setka."""
    data = _get_json(f"/Matches/widget/{locale}")
    return official_matches_to_frame(data)


def fetch_tournaments(
    match_date: str | date,
    locale: str = "en",
    day_period: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch official Setka tournaments/results for a date.

    `match_date` must be YYYY-MM-DD or a `datetime.date`.
    """
    if isinstance(match_date, date):
        date_text = match_date.isoformat()
    else:
        date_text = str(match_date)
    params: dict[str, Any] = {"date": date_text}
    if day_period is not None:
        params["dayPeriod"] = int(day_period)
    data = _get_json(f"/Tournaments/{locale}", params=params, timeout=30)
    return data or []


def flatten_tournaments(tournaments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tournament in tournaments or []:
        for match in tournament.get("matches", []) or []:
            row = dict(match)
            row.setdefault("tournamentId", tournament.get("id"))
            row.setdefault("tournamentName", tournament.get("name") or tournament.get("code"))
            row.setdefault("locationId", tournament.get("locationId"))
            row.setdefault("dayPeriodToken", tournament.get("dayPeriodToken"))
            rows.append(row)
    return rows


def _score_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def set_scores_text(set_scores: list[dict[str, Any]] | None) -> str:
    parts = []
    for set_score in set_scores or []:
        p1 = set_score.get("p1Score")
        p2 = set_score.get("p2Score")
        if p1 is not None and p2 is not None:
            parts.append(f"{p1}-{p2}")
    return ", ".join(parts)


def total_points_from_sets(set_scores: list[dict[str, Any]] | None) -> int | None:
    total = 0
    seen = False
    for set_score in set_scores or []:
        p1 = _score_int(set_score.get("p1Score"))
        p2 = _score_int(set_score.get("p2Score"))
        if p1 is not None and p2 is not None:
            total += p1 + p2
            seen = True
    return total if seen else None


def first_set_total(set_scores: list[dict[str, Any]] | None) -> int | None:
    if not set_scores:
        return None
    p1 = _score_int(set_scores[0].get("p1Score"))
    p2 = _score_int(set_scores[0].get("p2Score"))
    if p1 is None or p2 is None:
        return None
    return p1 + p2


def official_matches_to_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize Setka official match objects into one dataframe."""
    rows = []
    for match in matches or []:
        set_scores = match.get("setScores") or []
        winner = match.get("winner")
        active_player_id = match.get("activePlayerId")
        p1 = match.get("player1") or {}
        p2 = match.get("player2") or {}
        if active_player_id == p1.get("id"):
            active_player = player_name(p1)
        elif active_player_id == p2.get("id"):
            active_player = player_name(p2)
        else:
            active_player = ""
        rows.append(
            {
                "match_id": match.get("id") or match.get("matchId"),
                "tournament_id": match.get("tournamentId"),
                "tournament": match.get("tournamentName"),
                "location_id": match.get("locationId"),
                "day_period": match.get("dayPeriodToken"),
                "status_id": match.get("statusId"),
                "status": status_label(match.get("statusId")),
                "active_player_id": active_player_id,
                "active_player": active_player,
                "start_date_utc": match.get("startDate"),
                "player1": player_name(match.get("player1"))
                or f"{match.get('player1FirstName', '')} {match.get('player1LastName', '')}".strip(),
                "player2": player_name(match.get("player2"))
                or f"{match.get('player2FirstName', '')} {match.get('player2LastName', '')}".strip(),
                "player1_score": _score_int(match.get("player1Score")),
                "player2_score": _score_int(match.get("player2Score")),
                "winner": player_name(winner),
                "set_scores": set_scores_text(set_scores),
                "sets_played": len(set_scores),
                "total_points": total_points_from_sets(set_scores),
                "first_set_total": first_set_total(set_scores),
            }
        )
    return pd.DataFrame(rows)


def fetch_results_for_date(
    match_date: str | date,
    locale: str = "en",
    day_period: int | None = None,
    include_live_widget: bool = True,
) -> pd.DataFrame:
    tournaments = fetch_tournaments(match_date, locale=locale, day_period=day_period)
    frame = official_matches_to_frame(flatten_tournaments(tournaments))
    if include_live_widget:
        try:
            live = fetch_live_matches(locale=locale)
            if not live.empty:
                frame = pd.concat([frame, live], ignore_index=True)
        except Exception:
            pass
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["match_id"], keep="last")
    return frame.sort_values(["start_date_utc", "match_id"], na_position="last").reset_index(drop=True)


def add_lagos_time(frame: pd.DataFrame, source_col: str = "start_date_utc") -> pd.DataFrame:
    """Add `start_time_lagos` and `start_date_lagos` columns."""
    out = frame.copy()
    if out.empty or source_col not in out.columns:
        return out
    dt = pd.to_datetime(out[source_col], utc=True, errors="coerce").dt.tz_convert("Africa/Lagos")
    out["start_date_lagos"] = dt.dt.strftime("%Y-%m-%d")
    out["start_time_lagos"] = dt.dt.strftime("%H:%M")
    return out


def add_location_names(frame: pd.DataFrame, locations: dict[int, str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    if out.empty or "location_id" not in out.columns:
        return out
    locations = locations or location_map()
    out["location"] = out["location_id"].map(lambda x: locations.get(int(x), str(x)) if pd.notna(x) else "")
    return out
