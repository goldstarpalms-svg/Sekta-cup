from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _stats_row(player_stats: pd.DataFrame, player: str) -> dict[str, Any]:
    if player_stats is None or player_stats.empty or "player" not in player_stats:
        return {}
    rows = player_stats.loc[player_stats["player"] == player]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def calibrated_probability(probability: float, market: str = "generic") -> float:
    """Conservative probability calibration.

    In betting-style use, overconfident probabilities are dangerous. This layer
    shrinks probabilities toward 50%, with different strength per market.
    It is intentionally conservative until a larger calibrated ML model is saved.
    """
    p = max(0.01, min(0.99, float(probability)))
    shrink = {
        "winner": 0.92,
        "total": 0.86,
        "first_set": 0.84,
        "sets": 0.88,
        "generic": 0.88,
    }.get(str(market), 0.88)
    return max(0.01, min(0.99, 0.5 + (p - 0.5) * shrink))


def winner_component_agreement(pred: dict[str, Any], player_stats: pd.DataFrame) -> dict[str, Any]:
    """Compare independent winner signals and count agreement with final winner."""
    p1 = pred["player_a"]
    p2 = pred["player_b"]
    final = pred["predicted_winner"]
    a = _stats_row(player_stats, p1)
    b = _stats_row(player_stats, p2)

    components: dict[str, str | None] = {}
    components["elo"] = p1 if _safe_float(pred.get("elo_diff")) >= 0 else p2
    components["career_form"] = p1 if _safe_float(pred.get("player_a_win_rate"), 0.5) >= _safe_float(pred.get("player_b_win_rate"), 0.5) else p2
    components["recent_form"] = p1 if _safe_float(pred.get("player_a_recent_win_rate"), 0.5) >= _safe_float(pred.get("player_b_recent_win_rate"), 0.5) else p2
    components["first_set_strength"] = p1 if _safe_float(a.get("first_set_win_rate"), 0.5) >= _safe_float(b.get("first_set_win_rate"), 0.5) else p2
    components["point_dominance"] = p1 if _safe_float(pred.get("recent_point_diff_signal"), 0.0) >= 0 else p2
    if int(pred.get("h2h_matches", 0) or 0) >= 3:
        components["head_to_head"] = p1 if _safe_float(pred.get("h2h_player_a_win_rate"), 0.5) >= 0.5 else p2
    else:
        components["head_to_head"] = None

    available = {k: v for k, v in components.items() if v is not None}
    agrees = sum(1 for v in available.values() if v == final)
    total = len(available)
    score = agrees / total if total else 0.0
    return {
        "winner_agreement_score": score,
        "winner_agreement": f"{agrees}/{total}",
        "winner_agreement_components": ", ".join(f"{k}:{v}" for k, v in available.items()),
    }


def estimate_player_fatigue(
    player_log: pd.DataFrame,
    player: str,
    start_date_lagos: str | None,
    start_time_lagos: str | None,
) -> dict[str, Any]:
    """Estimate same-session fatigue from saved/history rows.

    This becomes stronger as daily official results are saved into the model context.
    """
    default = {
        "recent_matches_6h": 0,
        "recent_matches_12h": 0,
        "minutes_since_last_match": None,
        "last_match_sets": None,
        "fatigue_score": 0.0,
        "fatigue_label": "Unknown",
    }
    if player_log is None or player_log.empty or not start_date_lagos or not start_time_lagos:
        return default
    try:
        start_dt = pd.to_datetime(f"{start_date_lagos} {str(start_time_lagos)[:5]}")
    except Exception:
        return default
    log = player_log.loc[player_log["player"] == player].copy()
    if log.empty or "date_time" not in log:
        return default
    log["date_time"] = pd.to_datetime(log["date_time"], errors="coerce")
    log = log.loc[log["date_time"].notna() & (log["date_time"] < start_dt)].sort_values("date_time")
    if log.empty:
        return default
    recent_6h = log.loc[log["date_time"] >= start_dt - pd.Timedelta(hours=6)]
    recent_12h = log.loc[log["date_time"] >= start_dt - pd.Timedelta(hours=12)]
    last = log.iloc[-1]
    minutes_gap = (start_dt - last["date_time"]).total_seconds() / 60
    last_sets = _safe_float(last.get("sets_played"), 0.0)
    fatigue = 0.0
    fatigue += min(len(recent_6h), 4) * 0.12
    fatigue += min(len(recent_12h), 8) * 0.04
    if minutes_gap < 45:
        fatigue += 0.25
    elif minutes_gap < 90:
        fatigue += 0.12
    if last_sets >= 5:
        fatigue += 0.18
    elif last_sets >= 4:
        fatigue += 0.08
    fatigue = max(0.0, min(1.0, fatigue))
    label = "High" if fatigue >= 0.45 else "Medium" if fatigue >= 0.22 else "Low"
    return {
        "recent_matches_6h": int(len(recent_6h)),
        "recent_matches_12h": int(len(recent_12h)),
        "minutes_since_last_match": round(minutes_gap, 1),
        "last_match_sets": int(last_sets) if last_sets else None,
        "fatigue_score": fatigue,
        "fatigue_label": label,
    }


def match_fatigue_summary(
    player_log: pd.DataFrame,
    player1: str,
    player2: str,
    start_date_lagos: str | None,
    start_time_lagos: str | None,
) -> dict[str, Any]:
    f1 = estimate_player_fatigue(player_log, player1, start_date_lagos, start_time_lagos)
    f2 = estimate_player_fatigue(player_log, player2, start_date_lagos, start_time_lagos)
    diff = _safe_float(f1.get("fatigue_score")) - _safe_float(f2.get("fatigue_score"))
    max_label = "High" if max(_safe_float(f1.get("fatigue_score")), _safe_float(f2.get("fatigue_score"))) >= 0.45 else "Medium" if max(_safe_float(f1.get("fatigue_score")), _safe_float(f2.get("fatigue_score"))) >= 0.22 else "Low"
    return {
        "player1_fatigue": f1.get("fatigue_label"),
        "player2_fatigue": f2.get("fatigue_label"),
        "player1_recent_matches_6h": f1.get("recent_matches_6h"),
        "player2_recent_matches_6h": f2.get("recent_matches_6h"),
        "fatigue_diff_p1_minus_p2": diff,
        "match_fatigue_risk": max_label,
    }


def player_reliability_tags(player_stats: pd.DataFrame, player: str) -> str:
    row = _stats_row(player_stats, player)
    tags = []
    matches = _safe_float(row.get("matches"), 0)
    recent = _safe_float(row.get("recent_matches"), 0)
    if matches >= 300:
        tags.append("deep sample")
    elif matches >= 80:
        tags.append("good sample")
    else:
        tags.append("thin sample")
    if recent >= 15:
        tags.append("recent data")
    if abs(_safe_float(row.get("recent_avg_point_diff"), 0)) >= 4:
        tags.append("volatile form")
    return ", ".join(tags)


def market_confidence_label(
    probability: float,
    base_confidence: str,
    upset_risk: str,
    agreement_score: float,
    fatigue_risk: str,
) -> str:
    p = float(probability)
    score = p
    score += {"High": 0.06, "Medium": 0.02, "Low": -0.05}.get(base_confidence, 0)
    score += (agreement_score - 0.5) * 0.12
    score -= {"Low": 0.00, "Medium": 0.04, "High": 0.09}.get(upset_risk, 0.04)
    score -= {"Low": 0.00, "Medium": 0.02, "High": 0.05, "Unknown": 0.01}.get(fatigue_risk, 0.01)
    if score >= 0.72:
        return "Elite"
    if score >= 0.64:
        return "Strong"
    if score >= 0.57:
        return "Playable"
    return "Weak"
