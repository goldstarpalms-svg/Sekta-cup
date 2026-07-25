#!/usr/bin/env python3
"""CLI helper for official Setka upcoming-match predictions.

Example:
    python scripts/live_predictions.py --limit 20 --total-line 75.5 --first-set-line 18.5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.setka_core import build_context, load_raw_data, predict_match
from src.setka_live import add_lagos_time, add_location_names, fetch_nearest_matches, location_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict official Setka upcoming matches")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--total-line", type=float, default=75.5)
    parser.add_argument("--first-set-line", type=float, default=18.5)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    matches, leaderboard = load_raw_data()
    ctx = build_context(matches, leaderboard)
    upcoming = add_location_names(add_lagos_time(fetch_nearest_matches()), location_map()).head(args.limit)

    rows = []
    for _, row in upcoming.iterrows():
        pred = predict_match(
            row["player1"],
            row["player2"],
            ctx["player_stats"],
            ctx["matches"],
            ctx["global_stats"],
            first_set_line=args.first_set_line,
            total_points_line=args.total_line,
        )
        winner_prob = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
        total_pick = "Over" if pred["total_points_over_probability"] >= pred["total_points_under_probability"] else "Under"
        total_prob = max(pred["total_points_over_probability"], pred["total_points_under_probability"])
        first_pick = "Over" if pred["first_set_over_probability"] >= pred["first_set_under_probability"] else "Under"
        first_prob = max(pred["first_set_over_probability"], pred["first_set_under_probability"])
        rows.append(
            {
                "match_id": row["match_id"],
                "time_lagos": row["start_time_lagos"],
                "location": row.get("location"),
                "match": f"{row['player1']} vs {row['player2']}",
                "winner_pick": pred["predicted_winner"],
                "winner_probability": round(winner_prob * 100, 1),
                "total_pick": total_pick,
                "total_probability": round(total_prob * 100, 1),
                "expected_total_points": round(pred["expected_total_points"], 1),
                "first_set_pick": first_pick,
                "first_set_probability": round(first_prob * 100, 1),
                "expected_first_set_points": round(pred["expected_first_set_points"], 1),
                "confidence": pred["confidence"],
                "h2h_matches": pred["h2h_matches"],
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
