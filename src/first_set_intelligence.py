from __future__ import annotations

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


def first_set_matchups(matches: pd.DataFrame, player1: str, player2: str) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()
    mask = ((matches["player1"] == player1) & (matches["player2"] == player2)) | (
        (matches["player1"] == player2) & (matches["player2"] == player1)
    )
    out = matches.loc[mask].copy()
    if not out.empty and "date_time" in out.columns:
        out = out.sort_values("date_time", ascending=False)
    return out


def first_set_player_type(row: dict[str, Any]) -> str:
    avg = _safe_float(row.get("avg_first_set_total"), 18.7)
    recent = _safe_float(row.get("recent_avg_first_set_total"), avg)
    over = _safe_float(row.get("first_set_over_18_5_rate"), 0.5)
    recent_over = _safe_float(row.get("recent_first_set_over_18_5_rate"), over)
    std = _safe_float(row.get("std_first_set_total"), 3.2)
    score = (avg - 18.5) * 0.22 + (recent - 18.5) * 0.25 + (over - 0.5) * 1.1 + (recent_over - 0.5) * 0.9
    if std >= 4.2:
        volatility = "volatile"
    elif std <= 2.4:
        volatility = "stable"
    else:
        volatility = "normal"
    if score >= 0.42:
        return f"fast over starter / {volatility}"
    if score <= -0.42:
        return f"under starter / {volatility}"
    return f"neutral starter / {volatility}"


def first_set_intelligence(
    player1: str,
    player2: str,
    pred: dict[str, Any],
    player_stats: pd.DataFrame,
    matches: pd.DataFrame,
    line: float = 18.5,
) -> dict[str, Any]:
    """First-set-only intelligence for Setka O/U decisions.

    This does not try to force a pick. Its main value is to identify Strong
    Over/Under spots and mark weak/noisy matchups as Avoid.
    """
    a = _stats_row(player_stats, player1)
    b = _stats_row(player_stats, player2)
    h2h = first_set_matchups(matches, player1, player2)
    recent_h2h = h2h.head(10) if not h2h.empty else h2h

    expected = _safe_float(pred.get("expected_first_set_points"), 18.7)
    over_prob = _safe_float(pred.get("first_set_over_probability"), 0.5)
    under_prob = 1 - over_prob
    pick = "Over" if over_prob >= under_prob else "Under"
    pick_prob = max(over_prob, under_prob)

    a_over = _safe_float(a.get("first_set_over_18_5_rate"), 0.5)
    b_over = _safe_float(b.get("first_set_over_18_5_rate"), 0.5)
    a_recent_over = _safe_float(a.get("recent_first_set_over_18_5_rate"), a_over)
    b_recent_over = _safe_float(b.get("recent_first_set_over_18_5_rate"), b_over)
    combined_over_tendency = (a_over + b_over + a_recent_over + b_recent_over) / 4

    h2h_count = int(len(h2h))
    recent_h2h_count = int(len(recent_h2h)) if recent_h2h is not None else 0
    h2h_over = float(h2h["first_set_over_18_5"].mean()) if h2h_count else np.nan
    recent_h2h_over = float(recent_h2h["first_set_over_18_5"].mean()) if recent_h2h_count else np.nan
    h2h_avg = float(h2h["first_set_total"].mean()) if h2h_count else np.nan
    recent_h2h_avg = float(recent_h2h["first_set_total"].mean()) if recent_h2h_count else np.nan
    recent_h2h_scores = ""
    if recent_h2h_count and "first_set_p1_points" in recent_h2h.columns and "first_set_p2_points" in recent_h2h.columns:
        recent_h2h_scores = ", ".join(
            f"{int(r.first_set_p1_points)}-{int(r.first_set_p2_points)}"
            for r in recent_h2h.head(5).itertuples()
            if pd.notna(r.first_set_p1_points) and pd.notna(r.first_set_p2_points)
        )

    # Closeness: close winner probability + close first-set win rates + close recent H2H first-set margins.
    winner_prob = max(_safe_float(pred.get("player_a_win_probability"), 0.5), _safe_float(pred.get("player_b_win_probability"), 0.5))
    first_set_win_diff = abs(_safe_float(a.get("first_set_win_rate"), 0.5) - _safe_float(b.get("first_set_win_rate"), 0.5))
    if recent_h2h_count and {"first_set_p1_points", "first_set_p2_points"}.issubset(recent_h2h.columns):
        h2h_margin = float((recent_h2h["first_set_p1_points"] - recent_h2h["first_set_p2_points"]).abs().mean())
    else:
        h2h_margin = np.nan
    closeness_score = 1.0
    closeness_score -= min(abs(winner_prob - 0.5) * 1.8, 0.45)
    closeness_score -= min(first_set_win_diff * 1.4, 0.30)
    if not pd.isna(h2h_margin):
        closeness_score -= min(max(h2h_margin - 2.0, 0) * 0.05, 0.20)
    closeness_score = max(0.0, min(1.0, closeness_score))

    volatility = np.nanmean([
        _safe_float(a.get("std_first_set_total"), 3.2),
        _safe_float(b.get("std_first_set_total"), 3.2),
        float(h2h["first_set_total"].std()) if h2h_count > 1 else np.nan,
    ])
    volatility_label = "High" if volatility >= 4.1 else "Low" if volatility <= 2.4 else "Medium"

    signals: list[str] = []
    over_votes = 0
    under_votes = 0
    if expected >= line + 0.65:
        over_votes += 1; signals.append("expected points above line")
    elif expected <= line - 0.65:
        under_votes += 1; signals.append("expected points below line")
    if combined_over_tendency >= 0.54:
        over_votes += 1; signals.append("players trend over")
    elif combined_over_tendency <= 0.46:
        under_votes += 1; signals.append("players trend under")
    if recent_h2h_count >= 4 and not pd.isna(recent_h2h_over):
        if recent_h2h_over >= 0.60:
            over_votes += 1; signals.append("recent H2H over")
        elif recent_h2h_over <= 0.40:
            under_votes += 1; signals.append("recent H2H under")
    if closeness_score >= 0.62:
        over_votes += 1; signals.append("close first-set profile")
    elif closeness_score <= 0.38:
        under_votes += 1; signals.append("domination risk")
    if winner_prob >= 0.72 and first_set_win_diff >= 0.10:
        under_votes += 1; signals.append("strong starter mismatch")

    signal_total = over_votes + under_votes
    signal_agreement = max(over_votes, under_votes) / signal_total if signal_total else 0.0
    signal_pick = "Over" if over_votes >= under_votes else "Under"

    avoid_reasons: list[str] = []
    if pick_prob < 0.56:
        avoid_reasons.append("low model edge")
    if abs(expected - line) <= 0.35:
        avoid_reasons.append("expected points too close to line")
    if signal_total >= 2 and signal_agreement < 0.60:
        avoid_reasons.append("signals conflict")
    if volatility_label == "High" and pick_prob < 0.62:
        avoid_reasons.append("high first-set volatility")
    if h2h_count < 3 and pick_prob < 0.60:
        avoid_reasons.append("thin H2H for weak edge")

    if avoid_reasons:
        label = "Avoid"
    elif pick == signal_pick and pick_prob >= 0.61 and signal_agreement >= 0.67:
        label = f"Strong {pick}"
    elif pick == signal_pick and pick_prob >= 0.57:
        label = f"Lean {pick}"
    else:
        label = "Avoid"

    # Gambler lean: less strict than the professional filter, with recent H2H
    # first-set behavior weighted strongly. This is for visibility, not a
    # guarantee. It helps users who want action see the practical lean.
    h2h_component = recent_h2h_over if not pd.isna(recent_h2h_over) else h2h_over if not pd.isna(h2h_over) else 0.5
    expected_component = max(0.0, min(1.0, 0.5 + (expected - line) / 7.0))
    player_component = combined_over_tendency
    close_component = 0.5 + (closeness_score - 0.5) * 0.35
    h2h_weight = 0.42 if recent_h2h_count >= 3 else 0.25 if h2h_count >= 3 else 0.12
    remaining = 1 - h2h_weight
    gambler_over_score = (
        h2h_component * h2h_weight
        + expected_component * (remaining * 0.38)
        + player_component * (remaining * 0.34)
        + close_component * (remaining * 0.28)
    )
    gambler_pick = "Over" if gambler_over_score >= 0.5 else "Under"
    gambler_confidence = abs(gambler_over_score - 0.5) * 2
    if recent_h2h_count >= 3:
        gambler_reason = f"recent H2H {recent_h2h_count} games, O18.5 {recent_h2h_over:.0%}, scores {recent_h2h_scores or '-'}"
    elif h2h_count >= 3:
        gambler_reason = f"H2H {h2h_count} games, O18.5 {h2h_over:.0%}"
    else:
        gambler_reason = "thin H2H, using player tempo + expected points"
    if gambler_confidence >= 0.22:
        gambler_label = f"Gambler {gambler_pick}"
    elif gambler_confidence >= 0.12:
        gambler_label = f"Small lean {gambler_pick}"
    else:
        gambler_label = "Pass / coin-flip"

    return {
        "first_set_model_pick": pick,
        "first_set_model_probability": pick_prob,
        "first_set_label": label,
        "first_set_signal_pick": signal_pick,
        "first_set_signal_agreement": signal_agreement,
        "first_set_over_votes": over_votes,
        "first_set_under_votes": under_votes,
        "first_set_signals": ", ".join(signals) if signals else "neutral",
        "first_set_avoid_reasons": ", ".join(avoid_reasons),
        "first_set_closeness_score": closeness_score,
        "first_set_volatility": volatility_label,
        "combined_first_set_over_tendency": combined_over_tendency,
        "h2h_first_set_over_rate": h2h_over,
        "recent_h2h_first_set_over_rate": recent_h2h_over,
        "h2h_first_set_avg": h2h_avg,
        "recent_h2h_first_set_avg": recent_h2h_avg,
        "recent_h2h_first_set_scores": recent_h2h_scores,
        "first_set_gambler_label": gambler_label,
        "first_set_gambler_pick": gambler_pick,
        "first_set_gambler_over_score": gambler_over_score,
        "first_set_gambler_confidence": gambler_confidence,
        "first_set_gambler_reason": gambler_reason,
        "player1_first_set_type": first_set_player_type(a),
        "player2_first_set_type": first_set_player_type(b),
    }
