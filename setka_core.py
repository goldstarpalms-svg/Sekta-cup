from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

MATCHES_FILE = "Setka_June_2025_to_Now.csv"
LEADERBOARD_FILE = "setka_leaderboard.csv"


# -----------------------------
# Loading and preparation
# -----------------------------


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def load_raw_data(base_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two CSV files used by the app."""
    base = Path(base_dir) if base_dir else data_dir()
    matches_path = base / MATCHES_FILE
    leaderboard_path = base / LEADERBOARD_FILE

    if not matches_path.exists():
        raise FileNotFoundError(f"Missing match file: {matches_path}")
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard file: {leaderboard_path}")

    matches = pd.read_csv(matches_path)
    leaderboard = pd.read_csv(leaderboard_path)
    return matches, leaderboard


def parse_score_string(score: str) -> dict[str, float | int]:
    """Parse a Setka score string like '11-8;9-11;11-7'.

    Scores are stored from player1's perspective: each set is p1-p2.
    """
    p1_total = 0
    p2_total = 0
    p1_sets = 0
    p2_sets = 0
    first_p1 = np.nan
    first_p2 = np.nan
    sets_played = 0

    if pd.isna(score):
        return {
            "first_set_p1_points": np.nan,
            "first_set_p2_points": np.nan,
            "first_set_total": np.nan,
            "first_set_over_18_5": False,
            "total_points": np.nan,
            "p1_points": np.nan,
            "p2_points": np.nan,
            "sets_played": 0,
            "p1_sets_won": 0,
            "p2_sets_won": 0,
        }

    for i, part in enumerate(str(score).split(";")):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("-", 1)
        p1 = int(left)
        p2 = int(right)
        if i == 0:
            first_p1 = p1
            first_p2 = p2
        p1_total += p1
        p2_total += p2
        p1_sets += int(p1 > p2)
        p2_sets += int(p2 > p1)
        sets_played += 1

    first_total = first_p1 + first_p2 if not pd.isna(first_p1) else np.nan
    return {
        "first_set_p1_points": first_p1,
        "first_set_p2_points": first_p2,
        "first_set_total": first_total,
        "first_set_over_18_5": bool(first_total > 18.5) if not pd.isna(first_total) else False,
        "total_points": p1_total + p2_total,
        "p1_points": p1_total,
        "p2_points": p2_total,
        "sets_played": sets_played,
        "p1_sets_won": p1_sets,
        "p2_sets_won": p2_sets,
    }


def enrich_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Add parsed score metrics and clean datetimes."""
    required = {
        "date",
        "time",
        "competition",
        "player1",
        "player2",
        "winner",
        "set_scores",
        "source_match_id",
    }
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Match CSV is missing required columns: {sorted(missing)}")

    df = matches.copy()
    df["date_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )
    score_df = pd.DataFrame([parse_score_string(x) for x in df["set_scores"]], index=df.index)
    df = pd.concat([df, score_df], axis=1)
    df["p1_won"] = df["winner"].eq(df["player1"])
    df["first_set_p1_won"] = df["first_set_p1_points"] > df["first_set_p2_points"]
    df["first_set_p2_won"] = df["first_set_p2_points"] > df["first_set_p1_points"]
    return df


def make_player_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player per match."""
    m = matches.copy()

    common = {
        "date_time": m["date_time"],
        "date": m["date"],
        "time": m["time"],
        "competition": m["competition"],
        "winner": m["winner"],
        "set_scores": m["set_scores"],
        "source_match_id": m["source_match_id"],
        "total_points": m["total_points"],
        "first_set_total": m["first_set_total"],
        "first_set_over_18_5": m["first_set_over_18_5"],
        "sets_played": m["sets_played"],
    }

    p1 = pd.DataFrame(
        {
            **common,
            "player": m["player1"],
            "opponent": m["player2"],
            "side": "player1",
            "won": m["p1_won"].astype(bool),
            "points_for": m["p1_points"],
            "points_against": m["p2_points"],
            "sets_won": m["p1_sets_won"],
            "sets_lost": m["p2_sets_won"],
            "first_set_won": m["first_set_p1_won"].astype(bool),
        }
    )

    p2 = pd.DataFrame(
        {
            **common,
            "player": m["player2"],
            "opponent": m["player1"],
            "side": "player2",
            "won": (~m["p1_won"]).astype(bool),
            "points_for": m["p2_points"],
            "points_against": m["p1_points"],
            "sets_won": m["p2_sets_won"],
            "sets_lost": m["p1_sets_won"],
            "first_set_won": m["first_set_p2_won"].astype(bool),
        }
    )

    out = pd.concat([p1, p2], ignore_index=True)
    out["point_diff"] = out["points_for"] - out["points_against"]
    out["set_diff"] = out["sets_won"] - out["sets_lost"]
    return out


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace({0: np.nan})


def build_player_stats(
    player_log: pd.DataFrame, leaderboard: pd.DataFrame, recent_n: int = 20
) -> pd.DataFrame:
    """Build current player statistics from match history and final leaderboard Elo."""
    log = player_log.copy()
    log["won"] = log["won"].astype(bool)
    log["first_set_over_18_5"] = log["first_set_over_18_5"].astype(bool)
    log["first_set_won"] = log["first_set_won"].astype(bool)

    grouped = log.groupby("player", dropna=False)
    stats = grouped.agg(
        matches=("won", "size"),
        wins=("won", "sum"),
        avg_points_for=("points_for", "mean"),
        avg_points_against=("points_against", "mean"),
        avg_point_diff=("point_diff", "mean"),
        avg_sets_played=("sets_played", "mean"),
        avg_set_diff=("set_diff", "mean"),
        avg_total_points=("total_points", "mean"),
        std_total_points=("total_points", "std"),
        avg_first_set_total=("first_set_total", "mean"),
        std_first_set_total=("first_set_total", "std"),
        first_set_over_18_5_rate=("first_set_over_18_5", "mean"),
        first_set_win_rate=("first_set_won", "mean"),
        last_played=("date_time", "max"),
    ).reset_index()
    stats["losses"] = stats["matches"] - stats["wins"]
    stats["win_rate"] = _safe_div(stats["wins"], stats["matches"])

    log_sorted = log.sort_values(["player", "date_time", "source_match_id"])
    recent = log_sorted.groupby("player", group_keys=False).tail(recent_n)
    recent_stats = recent.groupby("player", dropna=False).agg(
        recent_matches=("won", "size"),
        recent_wins=("won", "sum"),
        recent_win_rate=("won", "mean"),
        recent_avg_total_points=("total_points", "mean"),
        recent_avg_first_set_total=("first_set_total", "mean"),
        recent_first_set_over_18_5_rate=("first_set_over_18_5", "mean"),
        recent_avg_point_diff=("point_diff", "mean"),
    ).reset_index()

    stats = stats.merge(recent_stats, on="player", how="left")

    lb = leaderboard.copy()
    lb["player"] = lb["player"].astype(str)
    lb = lb.rename(columns={"matches": "leaderboard_matches"})
    stats = stats.merge(lb[["player", "elo", "leaderboard_matches"]], on="player", how="outer")

    # Fill player rows that exist only in the leaderboard.
    fill_zero_cols = [
        "matches",
        "wins",
        "losses",
        "recent_matches",
        "recent_wins",
    ]
    for col in fill_zero_cols:
        if col in stats:
            stats[col] = stats[col].fillna(0).astype(int)

    rate_defaults = {
        "win_rate": 0.5,
        "recent_win_rate": stats.get("win_rate", pd.Series(0.5, index=stats.index)),
        "first_set_win_rate": 0.5,
        "first_set_over_18_5_rate": 0.5,
        "recent_first_set_over_18_5_rate": stats.get(
            "first_set_over_18_5_rate", pd.Series(0.5, index=stats.index)
        ),
    }
    for col, default in rate_defaults.items():
        if col in stats:
            stats[col] = stats[col].fillna(default)

    numeric_mean_cols = [
        "avg_points_for",
        "avg_points_against",
        "avg_point_diff",
        "avg_sets_played",
        "avg_set_diff",
        "avg_total_points",
        "std_total_points",
        "avg_first_set_total",
        "std_first_set_total",
        "recent_avg_total_points",
        "recent_avg_first_set_total",
        "recent_avg_point_diff",
    ]
    for col in numeric_mean_cols:
        if col in stats:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    # Fallback Elo: estimated from win-rate only when no leaderboard Elo exists.
    stats["elo_source"] = np.where(stats["elo"].notna(), "leaderboard", "estimated")
    estimated_elo = 1500 + (stats["win_rate"].fillna(0.5) - 0.5) * 450
    stats["elo"] = stats["elo"].fillna(estimated_elo).fillna(1500).round(1)
    stats["leaderboard_matches"] = stats["leaderboard_matches"].fillna(stats["matches"]).astype(int)

    stats = stats.sort_values(["elo", "matches"], ascending=[False, False]).reset_index(drop=True)
    return stats


def build_global_stats(matches: pd.DataFrame, player_stats: pd.DataFrame) -> dict[str, Any]:
    return {
        "match_count": int(len(matches)),
        "player_count": int(player_stats["player"].nunique()),
        "date_min": pd.to_datetime(matches["date_time"]).min(),
        "date_max": pd.to_datetime(matches["date_time"]).max(),
        "total_points_mean": float(matches["total_points"].mean()),
        "total_points_std": float(matches["total_points"].std()),
        "first_set_mean": float(matches["first_set_total"].mean()),
        "first_set_std": float(matches["first_set_total"].std()),
        "first_set_over_18_5_rate": float(matches["first_set_over_18_5"].mean()),
        "avg_sets_played": float(matches["sets_played"].mean()),
    }


def build_context(
    matches: pd.DataFrame, leaderboard: pd.DataFrame, recent_n: int = 20
) -> dict[str, Any]:
    enriched = enrich_matches(matches)
    player_log = make_player_match_log(enriched)
    player_stats = build_player_stats(player_log, leaderboard, recent_n=recent_n)
    global_stats = build_global_stats(enriched, player_stats)
    return {
        "matches": enriched,
        "player_log": player_log,
        "player_stats": player_stats,
        "global_stats": global_stats,
    }


# -----------------------------
# Head-to-head and prediction
# -----------------------------


def get_head_to_head(
    matches: pd.DataFrame, player_a: str, player_b: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return head-to-head summary and matching rows sorted newest first."""
    mask = (
        ((matches["player1"] == player_a) & (matches["player2"] == player_b))
        | ((matches["player1"] == player_b) & (matches["player2"] == player_a))
    )
    h2h = matches.loc[mask].copy()
    if h2h.empty:
        summary = {
            "matches": 0,
            "player_a_wins": 0,
            "player_b_wins": 0,
            "player_a_win_rate": 0.5,
            "avg_total_points": np.nan,
            "std_total_points": np.nan,
            "avg_first_set_total": np.nan,
            "std_first_set_total": np.nan,
            "first_set_over_18_5_rate": np.nan,
            "last_played": pd.NaT,
        }
        return summary, h2h

    a_won = ((h2h["player1"] == player_a) & h2h["p1_won"]) | (
        (h2h["player2"] == player_a) & (~h2h["p1_won"])
    )
    h2h["selected_player_won"] = a_won
    h2h["matchup"] = player_a + " vs " + player_b

    a_wins = int(a_won.sum())
    total = int(len(h2h))
    summary = {
        "matches": total,
        "player_a_wins": a_wins,
        "player_b_wins": total - a_wins,
        "player_a_win_rate": a_wins / total if total else 0.5,
        "avg_total_points": float(h2h["total_points"].mean()),
        "std_total_points": float(h2h["total_points"].std()) if total > 1 else np.nan,
        "avg_first_set_total": float(h2h["first_set_total"].mean()),
        "std_first_set_total": float(h2h["first_set_total"].std()) if total > 1 else np.nan,
        "first_set_over_18_5_rate": float(h2h["first_set_over_18_5"].mean()),
        "last_played": h2h["date_time"].max(),
    }
    return summary, h2h.sort_values("date_time", ascending=False)


def _clamp(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return float((low + high) / 2)
    return float(max(low, min(high, value)))


def _logit(p: float) -> float:
    p = _clamp(p, 0.001, 0.999)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def _weighted_mean(values: Iterable[tuple[float | int | None, float]], default: float) -> float:
    num = 0.0
    den = 0.0
    for value, weight in values:
        if value is None or pd.isna(value) or weight <= 0:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return float(num / den) if den else float(default)


def _normal_over_probability(mean: float, line: float, std: float) -> float:
    std = max(float(std), 0.1)
    z = (float(line) - float(mean)) / std
    probability = 0.5 * math.erfc(z / math.sqrt(2))
    return _clamp(probability, 0.02, 0.98)


def _stats_lookup(player_stats: pd.DataFrame, player: str) -> dict[str, Any]:
    if "player" not in player_stats:
        return {}
    rows = player_stats.loc[player_stats["player"] == player]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _reliability(match_count: float | int | None, full_at: int = 80) -> float:
    if match_count is None or pd.isna(match_count):
        return 0.0
    return _clamp(float(match_count) / full_at, 0.0, 1.0)


def _value(row: dict[str, Any], key: str, default: float) -> float:
    val = row.get(key, default)
    if val is None or pd.isna(val):
        return default
    return float(val)


def _confidence_label(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def predict_match(
    player_a: str,
    player_b: str,
    player_stats: pd.DataFrame,
    matches: pd.DataFrame,
    global_stats: dict[str, Any],
    first_set_line: float = 18.5,
    total_points_line: float = 75.5,
) -> dict[str, Any]:
    """Blend Elo, form, career numbers, H2H, and point totals into one prediction.

    This is intentionally transparent and lightweight for a dashboard. It is not a
    guaranteed betting model.
    """
    if player_a == player_b:
        raise ValueError("Choose two different players.")

    a = _stats_lookup(player_stats, player_a)
    b = _stats_lookup(player_stats, player_b)
    h2h, h2h_rows = get_head_to_head(matches, player_a, player_b)

    a_matches = _value(a, "matches", 0)
    b_matches = _value(b, "matches", 0)
    a_rel = _reliability(a_matches)
    b_rel = _reliability(b_matches)
    h2h_rel = _reliability(h2h["matches"], full_at=20)

    elo_a = _value(a, "elo", 1500)
    elo_b = _value(b, "elo", 1500)
    elo_probability = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    career_diff = _value(a, "win_rate", 0.5) - _value(b, "win_rate", 0.5)
    recent_diff = _value(a, "recent_win_rate", _value(a, "win_rate", 0.5)) - _value(
        b, "recent_win_rate", _value(b, "win_rate", 0.5)
    )
    first_set_diff = _value(a, "first_set_win_rate", 0.5) - _value(
        b, "first_set_win_rate", 0.5
    )
    h2h_diff = h2h["player_a_win_rate"] - 0.5

    # Elo is the anchor; form and H2H are smaller adjustments.
    score = _logit(elo_probability)
    score += 0.70 * career_diff * min(a_rel, b_rel)
    score += 0.95 * recent_diff * min(a_rel, b_rel)
    score += 0.35 * first_set_diff * min(a_rel, b_rel)
    score += 0.90 * h2h_diff * h2h_rel
    player_a_win_probability = _clamp(_sigmoid(score), 0.03, 0.97)
    player_b_win_probability = 1 - player_a_win_probability

    global_first_mean = float(global_stats.get("first_set_mean", 18.7))
    global_first_std = float(global_stats.get("first_set_std", 3.2))
    global_first_over = float(global_stats.get("first_set_over_18_5_rate", 0.5))
    global_total_mean = float(global_stats.get("total_points_mean", 75.3))
    global_total_std = float(global_stats.get("total_points_std", 16.6))

    # Expected first-set points.
    expected_first = _weighted_mean(
        [
            (global_first_mean, 1.00),
            (_value(a, "avg_first_set_total", global_first_mean), 1.25 * a_rel),
            (_value(b, "avg_first_set_total", global_first_mean), 1.25 * b_rel),
            (_value(a, "recent_avg_first_set_total", global_first_mean), 0.90 * a_rel),
            (_value(b, "recent_avg_first_set_total", global_first_mean), 0.90 * b_rel),
            (h2h["avg_first_set_total"], 1.80 * h2h_rel),
        ],
        default=global_first_mean,
    )
    # Closer matches tend to push point totals slightly upward.
    expected_first += (0.5 - abs(player_a_win_probability - 0.5)) * 1.00
    expected_first = _clamp(expected_first, 11.0, 35.0)

    first_std = _weighted_mean(
        [
            (global_first_std, 1.0),
            (_value(a, "std_first_set_total", global_first_std), 0.6 * a_rel),
            (_value(b, "std_first_set_total", global_first_std), 0.6 * b_rel),
            (h2h["std_first_set_total"], 1.0 * h2h_rel),
        ],
        default=global_first_std,
    )
    first_line_probability = _normal_over_probability(expected_first, first_set_line, first_std)

    if abs(first_set_line - 18.5) < 1e-9:
        empirical_over = _weighted_mean(
            [
                (global_first_over, 1.0),
                (_value(a, "first_set_over_18_5_rate", global_first_over), 1.2 * a_rel),
                (_value(b, "first_set_over_18_5_rate", global_first_over), 1.2 * b_rel),
                (
                    _value(a, "recent_first_set_over_18_5_rate", global_first_over),
                    0.8 * a_rel,
                ),
                (
                    _value(b, "recent_first_set_over_18_5_rate", global_first_over),
                    0.8 * b_rel,
                ),
                (h2h["first_set_over_18_5_rate"], 1.5 * h2h_rel),
            ],
            default=global_first_over,
        )
        first_set_over_probability = _clamp(
            0.55 * empirical_over + 0.45 * first_line_probability, 0.03, 0.97
        )
    else:
        first_set_over_probability = first_line_probability

    # Expected whole-match points.
    expected_total = _weighted_mean(
        [
            (global_total_mean, 1.00),
            (_value(a, "avg_total_points", global_total_mean), 1.25 * a_rel),
            (_value(b, "avg_total_points", global_total_mean), 1.25 * b_rel),
            (_value(a, "recent_avg_total_points", global_total_mean), 0.90 * a_rel),
            (_value(b, "recent_avg_total_points", global_total_mean), 0.90 * b_rel),
            (h2h["avg_total_points"], 1.80 * h2h_rel),
        ],
        default=global_total_mean,
    )
    # Close contests generally go 4/5 sets more often.
    expected_total += ((0.5 - abs(player_a_win_probability - 0.5)) * 6.0) - 1.2
    expected_total = _clamp(expected_total, 33.0, 135.0)

    total_std = _weighted_mean(
        [
            (global_total_std, 1.0),
            (_value(a, "std_total_points", global_total_std), 0.6 * a_rel),
            (_value(b, "std_total_points", global_total_std), 0.6 * b_rel),
            (h2h["std_total_points"], 1.0 * h2h_rel),
        ],
        default=global_total_std,
    )
    total_over_probability = _normal_over_probability(expected_total, total_points_line, total_std)

    confidence_score = 0.0
    confidence_score += 30 * min(a_rel, b_rel)
    confidence_score += 20 * h2h_rel
    confidence_score += 20 if a.get("elo_source") == "leaderboard" and b.get("elo_source") == "leaderboard" else 8
    confidence_score += 15 * min(_reliability(_value(a, "recent_matches", 0), 20), _reliability(_value(b, "recent_matches", 0), 20))
    confidence_score += 15 if h2h_rows.shape[0] >= 3 else 5
    confidence = _confidence_label(confidence_score)

    return {
        "player_a": player_a,
        "player_b": player_b,
        "player_a_win_probability": player_a_win_probability,
        "player_b_win_probability": player_b_win_probability,
        "predicted_winner": player_a if player_a_win_probability >= 0.5 else player_b,
        "elo_probability": elo_probability,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "player_a_matches": int(a_matches),
        "player_b_matches": int(b_matches),
        "player_a_win_rate": _value(a, "win_rate", 0.5),
        "player_b_win_rate": _value(b, "win_rate", 0.5),
        "player_a_recent_win_rate": _value(a, "recent_win_rate", _value(a, "win_rate", 0.5)),
        "player_b_recent_win_rate": _value(b, "recent_win_rate", _value(b, "win_rate", 0.5)),
        "h2h_matches": int(h2h["matches"]),
        "h2h_player_a_wins": int(h2h["player_a_wins"]),
        "h2h_player_b_wins": int(h2h["player_b_wins"]),
        "h2h_player_a_win_rate": float(h2h["player_a_win_rate"]),
        "expected_first_set_points": expected_first,
        "first_set_line": float(first_set_line),
        "first_set_over_probability": first_set_over_probability,
        "first_set_under_probability": 1 - first_set_over_probability,
        "expected_total_points": expected_total,
        "total_points_line": float(total_points_line),
        "total_points_over_probability": total_over_probability,
        "total_points_under_probability": 1 - total_over_probability,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "h2h_table": h2h_rows,
    }


def comparison_table(player_stats: pd.DataFrame, player_a: str, player_b: str) -> pd.DataFrame:
    cols = [
        "player",
        "elo",
        "matches",
        "win_rate",
        "recent_win_rate",
        "avg_total_points",
        "avg_first_set_total",
        "first_set_over_18_5_rate",
        "avg_point_diff",
        "last_played",
    ]
    table = player_stats.loc[player_stats["player"].isin([player_a, player_b]), cols].copy()
    table["win_rate"] = table["win_rate"].map(lambda x: f"{x:.1%}")
    table["recent_win_rate"] = table["recent_win_rate"].map(lambda x: f"{x:.1%}")
    table["first_set_over_18_5_rate"] = table["first_set_over_18_5_rate"].map(
        lambda x: f"{x:.1%}"
    )
    for c in ["avg_total_points", "avg_first_set_total", "avg_point_diff"]:
        table[c] = table[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    table = table.rename(
        columns={
            "elo": "Elo",
            "matches": "Matches",
            "win_rate": "Win rate",
            "recent_win_rate": "Recent win rate",
            "avg_total_points": "Avg match points",
            "avg_first_set_total": "Avg 1st-set points",
            "first_set_over_18_5_rate": "1st set O18.5 rate",
            "avg_point_diff": "Avg point diff",
            "last_played": "Last played",
        }
    )
    return table


def format_percent(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}"
