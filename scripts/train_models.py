from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_pipeline import metrics_table, save_model_bundle, train_model_bundle
from src.setka_core import load_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Setka ML prediction models.")
    parser.add_argument("--data-dir", default="data", help="Folder containing the two CSV files.")
    parser.add_argument("--output", default="models/setka_ml_bundle.joblib", help="Output model bundle path.")
    parser.add_argument(
        "--algorithm",
        choices=["auto", "xgboost", "sklearn"],
        default="auto",
        help="Use XGBoost if installed, or scikit-learn HistGradientBoosting.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Chronological holdout fraction.")
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=0,
        help="Optional cap on orientation rows used for fitting; 0 means all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches, _leaderboard = load_raw_data(Path(args.data_dir))
    bundle = train_model_bundle(
        matches,
        algorithm=args.algorithm,
        test_size=args.test_size,
        max_training_rows=args.max_training_rows or None,
    )
    output_path = save_model_bundle(bundle, args.output)

    print(f"Saved model bundle: {output_path}")
    print(f"Algorithm: {bundle['algorithm']}")
    print(f"Rows total: {bundle['rows_total']:,}")
    print(f"Rows used: {bundle['rows_used_for_training']:,}")
    print(f"Train rows: {bundle['train_rows']:,}")
    print(f"Test rows: {bundle['test_rows']:,}")
    print("\nMetrics:")
    print(metrics_table(bundle).to_string(index=False))


if __name__ == "__main__":
    main()
