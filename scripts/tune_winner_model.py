from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_pipeline import FEATURE_COLUMNS, TARGET_WIN, _predict_probability, build_feature_frame
from src.setka_core import load_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the Setka winner model with Optuna.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--algorithm", choices=["sklearn", "xgboost"], default="sklearn")
    parser.add_argument("--max-training-rows", type=int, default=50_000)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--output", default="models/optuna_winner_trials.csv")
    return parser.parse_args()


def make_model(trial, algorithm: str):
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 600),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 100, 450),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 0.5, log=True),
        random_state=42,
    )


def main() -> None:
    args = parse_args()
    try:
        import optuna
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise SystemExit(
            "Optuna tuning requires optional dependencies. Install with:\n"
            "pip install -r requirements-optional.txt"
        ) from exc

    matches, _leaderboard = load_raw_data(args.data_dir)
    features, _state = build_feature_frame(matches, max_rows=args.max_training_rows)
    features = features.sort_values(["date_time", "source_match_id", "orientation"])

    split_idx = int(len(features) * 0.8)
    train_df = features.iloc[:split_idx]
    valid_df = features.iloc[split_idx:]
    x_train = train_df[FEATURE_COLUMNS].astype(float)
    y_train = train_df[TARGET_WIN].astype(int).to_numpy()
    x_valid = valid_df[FEATURE_COLUMNS].astype(float)
    y_valid = valid_df[TARGET_WIN].astype(int).to_numpy()

    def objective(trial):
        model = make_model(trial, args.algorithm)
        model.fit(x_train, y_train)
        prob = _predict_probability(model, x_valid)
        return float(roc_auc_score(y_valid, prob))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)

    trials_df = study.trials_dataframe()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(output, index=False)
    print("Saved trials:", output)


if __name__ == "__main__":
    main()
