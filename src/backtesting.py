from __future__ import annotations

from typing import Any

import pandas as pd

from src.setka_core import build_context, enrich_matches, predict_match


def run_holdout_backtest(
    matches_raw: pd.DataFrame,
    leaderboard: pd.DataFrame,
    test_rows: int = 750,
    first_set_line: float = 18.5,
    total_points_line: float = 75.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Time-split rule-model backtest.

    Uses historical rows before the holdout window to build player stats, then
    predicts the latest `test_rows` matches. This avoids using the tested match
    result itself in the player statistics.
    """
    enriched_all = enrich_matches(matches_raw).sort_values(["date_time", "source_match_id"]).reset_index(drop=True)
    test_rows = int(max(100, min(test_rows, len(enriched_all) // 2)))
    train_raw = matches_raw.loc[enriched_all.index[:-test_rows]].copy()
    test = enriched_all.tail(test_rows).copy()

    ctx = build_context(train_raw, leaderboard)
    rows = []
    for _, match in test.iterrows():
        try:
            pred = predict_match(
                match["player1"],
                match["player2"],
                ctx["player_stats"],
                ctx["matches"],
                ctx["global_stats"],
                first_set_line=first_set_line,
                total_points_line=total_points_line,
            )
        except Exception:
            continue
        winner_prob = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
        total_pick = "Over" if pred["total_points_over_probability"] >= pred["total_points_under_probability"] else "Under"
        total_prob = max(pred["total_points_over_probability"], pred["total_points_under_probability"])
        first_pick = "Over" if pred["first_set_over_probability"] >= pred["first_set_under_probability"] else "Under"
        first_prob = max(pred["first_set_over_probability"], pred["first_set_under_probability"])
        actual_total_pick = "Over" if float(match["total_points"]) > total_points_line else "Under"
        actual_first_pick = "Over" if float(match["first_set_total"]) > first_set_line else "Under"
        rows.append(
            {
                "date_time": match["date_time"],
                "match_id": match.get("source_match_id"),
                "match": f"{match['player1']} vs {match['player2']}",
                "predicted_winner": pred["predicted_winner"],
                "actual_winner": match["winner"],
                "winner_correct": pred["predicted_winner"] == match["winner"],
                "winner_probability": winner_prob,
                "total_pick": total_pick,
                "actual_total_pick": actual_total_pick,
                "total_correct": total_pick == actual_total_pick,
                "total_probability": total_prob,
                "first_set_pick": first_pick,
                "actual_first_set_pick": actual_first_pick,
                "first_set_correct": first_pick == actual_first_pick,
                "first_set_probability": first_prob,
                "confidence": pred.get("confidence"),
                "confidence_score": pred.get("confidence_score"),
                "upset_risk": pred.get("upset_risk"),
                "h2h_matches": pred.get("h2h_matches"),
            }
        )
    df = pd.DataFrame(rows)
    metrics = summarize_backtest(df)
    return df, metrics


def summarize_backtest(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "winner_accuracy": float(df["winner_correct"].mean()),
        "total_accuracy": float(df["total_correct"].mean()),
        "first_set_accuracy": float(df["first_set_correct"].mean()),
    }
    for threshold in [0.55, 0.58, 0.60, 0.65, 0.70]:
        subset = df.loc[df["winner_probability"] >= threshold]
        metrics[f"winner_acc_{threshold:.2f}"] = float(subset["winner_correct"].mean()) if not subset.empty else None
        metrics[f"winner_count_{threshold:.2f}"] = int(len(subset))
        subset = df.loc[df["total_probability"] >= threshold]
        metrics[f"total_acc_{threshold:.2f}"] = float(subset["total_correct"].mean()) if not subset.empty else None
        metrics[f"total_count_{threshold:.2f}"] = int(len(subset))
        subset = df.loc[df["first_set_probability"] >= threshold]
        metrics[f"first_acc_{threshold:.2f}"] = float(subset["first_set_correct"].mean()) if not subset.empty else None
        metrics[f"first_count_{threshold:.2f}"] = int(len(subset))
    return metrics


def threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, prob_col, correct_col in [
        ("Winner", "winner_probability", "winner_correct"),
        ("Total", "total_probability", "total_correct"),
        ("1st Set", "first_set_probability", "first_set_correct"),
    ]:
        for threshold in [0.50, 0.55, 0.58, 0.60, 0.65, 0.70]:
            subset = df.loc[df[prob_col] >= threshold]
            rows.append(
                {
                    "market": market,
                    "min_probability": threshold,
                    "picks": len(subset),
                    "accuracy": subset[correct_col].mean() if not subset.empty else None,
                }
            )
    return pd.DataFrame(rows)
