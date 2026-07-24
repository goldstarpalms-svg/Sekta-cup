from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "eu,uk,us"
DEFAULT_MARKETS = "h2h,totals"


class OddsAPIError(RuntimeError):
    """Raised when The Odds API returns an error response."""


def get_api_key(default: str | None = None) -> str | None:
    """Read The Odds API key from environment.

    In Streamlit Cloud, you can set this as a secret named THE_ODDS_API_KEY.
    In local development, export THE_ODDS_API_KEY=your_key_here.
    """
    return os.getenv("THE_ODDS_API_KEY") or default


def _request(path: str, api_key: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    if not api_key:
        raise OddsAPIError("Missing The Odds API key. Set THE_ODDS_API_KEY first.")
    params = dict(params or {})
    params["apiKey"] = api_key
    url = f"{BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=30)
    quota_headers = {
        "requests_remaining": response.headers.get("x-requests-remaining", ""),
        "requests_used": response.headers.get("x-requests-used", ""),
        "requests_last": response.headers.get("x-requests-last", ""),
    }
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise OddsAPIError(f"Odds API error {response.status_code}: {detail}")
    return response.json(), quota_headers


def list_sports(api_key: str, all_sports: bool = False) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return sports available for the account/key."""
    data, quota = _request("/sports/", api_key, params={"all": str(all_sports).lower()})
    df = pd.DataFrame(data)
    if not df.empty:
        cols = [c for c in ["key", "group", "title", "description", "active", "has_outrights"] if c in df]
        df = df[cols].sort_values(["group", "title"], na_position="last")
    return df, quota


def fetch_odds(
    api_key: str,
    sport_key: str,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    odds_format: str = "decimal",
    date_format: str = "iso",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch odds for a sport key from The Odds API.

    Example sport keys vary by account and availability. Use list_sports() in
    the app to discover the exact key for table tennis if it is available.
    """
    path = f"/sports/{sport_key}/odds/"
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    return _request(path, api_key, params=params)


def normalize_odds_events(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten Odds API event/bookmaker/market/outcome JSON into a table."""
    rows: list[dict[str, Any]] = []
    for event in events:
        event_base = {
            "event_id": event.get("id"),
            "sport_key": event.get("sport_key"),
            "sport_title": event.get("sport_title"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
        }
        for bookmaker in event.get("bookmakers", []) or []:
            bookmaker_base = {
                "bookmaker_key": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "last_update": bookmaker.get("last_update"),
            }
            for market in bookmaker.get("markets", []) or []:
                market_base = {
                    "market_key": market.get("key"),
                    "market_last_update": market.get("last_update"),
                }
                for outcome in market.get("outcomes", []) or []:
                    rows.append(
                        {
                            **event_base,
                            **bookmaker_base,
                            **market_base,
                            "outcome_name": outcome.get("name"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        }
                    )
    return pd.DataFrame(rows)


def decimal_to_implied_probability(price: float | int | None) -> float | None:
    if price is None:
        return None
    try:
        price = float(price)
    except Exception:
        return None
    if price <= 1:
        return None
    return 1 / price


def american_to_implied_probability(price: float | int | None) -> float | None:
    if price is None:
        return None
    try:
        price = float(price)
    except Exception:
        return None
    if price > 0:
        return 100 / (price + 100)
    if price < 0:
        return (-price) / ((-price) + 100)
    return None


def add_implied_probabilities(df: pd.DataFrame, odds_format: str = "decimal") -> pd.DataFrame:
    out = df.copy()
    if "price" in out:
        if odds_format == "american":
            out["implied_probability"] = out["price"].map(american_to_implied_probability)
        else:
            out["implied_probability"] = out["price"].map(decimal_to_implied_probability)
    return out
