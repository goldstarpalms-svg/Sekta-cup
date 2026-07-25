#!/usr/bin/env python3
"""CLI helper for official Setka results/live-score checking.

Example:
    python scripts/check_results.py --date 2026-07-25 --search Filip
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.setka_live import add_lagos_time, add_location_names, fetch_results_for_date, location_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Check official Setka results")
    parser.add_argument("--date", default=pd.Timestamp.now(tz="Africa/Lagos").date().isoformat())
    parser.add_argument("--period", choices=["all", "morning", "evening", "night"], default="all")
    parser.add_argument("--search", default="", help="Filter by player/location/match id")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    period_map = {"all": None, "morning": 1, "evening": 2, "night": 3}
    frame = fetch_results_for_date(args.date, day_period=period_map[args.period])
    frame = add_location_names(add_lagos_time(frame), location_map())
    if frame.empty:
        print("No results returned.")
        return 0
    frame["match"] = frame["player1"] + " vs " + frame["player2"]
    if args.search:
        needle = args.search.lower()
        frame = frame.loc[frame.apply(lambda r: needle in " ".join(str(v).lower() for v in r.values), axis=1)]
    cols = [
        "start_time_lagos",
        "location",
        "status",
        "match",
        "player1_score",
        "player2_score",
        "winner",
        "set_scores",
        "total_points",
        "first_set_total",
        "match_id",
    ]
    out = frame[[c for c in cols if c in frame.columns]]
    print(out.to_string(index=False))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
