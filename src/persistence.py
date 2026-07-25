from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

APP_STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "app_state"
STRONG_PICKS_FILE = APP_STATE_DIR / "strong_picks.csv"
DAILY_RESULTS_FILE = APP_STATE_DIR / "daily_results.csv"


def ensure_state_dir() -> Path:
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return APP_STATE_DIR


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_table(frame: pd.DataFrame, path: str | Path) -> Path:
    ensure_state_dir()
    path = Path(path)
    frame.to_csv(path, index=False)
    return path


def append_unique(
    new_rows: pd.DataFrame,
    path: str | Path,
    subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    ensure_state_dir()
    existing = load_table(path)
    if new_rows is None or new_rows.empty:
        return existing
    combined = pd.concat([existing, new_rows.copy()], ignore_index=True)
    if subset:
        subset_cols = [c for c in subset if c in combined.columns]
        if subset_cols:
            combined = combined.drop_duplicates(subset=subset_cols, keep="last")
    save_table(combined, path)
    return combined


def reset_table(path: str | Path) -> None:
    path = Path(path)
    ensure_state_dir()
    if path.exists():
        path.unlink()


def load_strong_picks() -> pd.DataFrame:
    return load_table(STRONG_PICKS_FILE)


def save_strong_picks(frame: pd.DataFrame) -> pd.DataFrame:
    return append_unique(frame, STRONG_PICKS_FILE, subset=["match_id", "best_market", "best_pick"])


def reset_strong_picks() -> None:
    reset_table(STRONG_PICKS_FILE)


def load_daily_results() -> pd.DataFrame:
    return load_table(DAILY_RESULTS_FILE)


def save_daily_results(frame: pd.DataFrame) -> pd.DataFrame:
    return append_unique(frame, DAILY_RESULTS_FILE, subset=["match_id"])


def reset_daily_results() -> None:
    reset_table(DAILY_RESULTS_FILE)


def official_results_to_match_history(results: pd.DataFrame) -> pd.DataFrame:
    """Convert saved official Setka result rows into the app's historical match schema.

    This lets newly synced official results be appended to the local training/stat
    context on the next app load. Only finished rows with winner and set scores
    are converted.
    """
    if results is None or results.empty:
        return pd.DataFrame(
            columns=["date", "time", "competition", "player1", "player2", "winner", "set_scores", "source_match_id"]
        )
    df = results.copy()
    if "status" in df.columns:
        df = df.loc[df["status"].astype(str).isin(["Finished", "Technical"])]
    required = {"match_id", "player1", "player2", "winner", "set_scores"}
    if not required.issubset(df.columns):
        return pd.DataFrame(
            columns=["date", "time", "competition", "player1", "player2", "winner", "set_scores", "source_match_id"]
        )
    df = df.loc[df["winner"].notna() & df["set_scores"].notna() & (df["set_scores"].astype(str).str.len() > 0)].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["date", "time", "competition", "player1", "player2", "winner", "set_scores", "source_match_id"]
        )

    if "start_date_lagos" not in df.columns or df["start_date_lagos"].isna().all():
        dt = pd.to_datetime(df.get("start_date_utc"), utc=True, errors="coerce").dt.tz_convert("Africa/Lagos")
        df["date"] = dt.dt.strftime("%Y-%m-%d")
        df["time"] = dt.dt.strftime("%H:%M")
    else:
        df["date"] = df["start_date_lagos"].astype(str)
        df["time"] = df.get("start_time_lagos", "00:00")

    out = pd.DataFrame(
        {
            "date": df["date"],
            "time": df["time"].astype(str).str.slice(0, 5),
            "competition": df.get("tournament", "Official Setka"),
            "player1": df["player1"],
            "player2": df["player2"],
            "winner": df["winner"],
            "set_scores": df["set_scores"].astype(str).str.replace(", ", ";", regex=False).str.replace(",", ";", regex=False),
            "source_match_id": pd.to_numeric(df["match_id"], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["source_match_id", "player1", "player2", "winner", "set_scores"])
    out["source_match_id"] = out["source_match_id"].astype(int)
    return out.drop_duplicates(subset=["source_match_id"], keep="last")
