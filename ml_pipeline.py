from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .setka_core import enrich_matches

FEATURE_COLUMNS = [
    "a_elo",
    "b_elo",
    "elo_diff",
    "elo_probability",
    "a_matches",
    "b_matches",
    "match_count_diff",
    "a_win_rate",
    "b_win_rate",
    "win_rate_diff",
    "a_recent_win_rate",
    "b_recent_win_rate",
    "recent_win_rate_diff",
    "a_first_set_win_rate",
    "b_first_set_win_rate",
    "first_set_win_rate_diff",
    "a_point_diff_avg",
    "b_point_diff_avg",
    "point_diff_diff",
    "a_avg_total_points",
    "b_avg_total_points",
    "avg_total_points_mean",
    "avg_total_points_diff",
    "a_recent_avg_total_points",
    "b_recent_avg_total_points",
    "recent_avg_total_points_mean",
    "a_avg_first_set_total",
    "b_avg_first_set_total",
    "avg_first_set_total_mean",
    "a_recent_avg_first_set_total",
    "b_recent_avg_first_set_total",
    "recent_avg_first_set_total_mean",
    "a_first_over_rate",
    "b_first_over_rate",
    "first_over_rate_mean",
    "first_over_rate_diff",
    "a_recent_first_over_rate",
    "b_recent_first_over_rate",
    "recent_first_over_rate_mean",
    "h2h_matches",
    "h2h_a_win_rate",
    "h2h_a_win_diff",
    "h2h_avg_total_points",
    "h2h_avg_first_set_total",
    "h2h_first_over_rate",
]

TARGET_WIN = "target_a_win"
TARGET_FIRST_OVER = "target_first_set_over_18_5"
TARGET_TOTAL_POINTS = "target_total_points"
TARGET_FIRST_SET_POINTS = "target_first_set_points"


# -----------------------------
# Rolling feature state
# -----------------------------


def _rate(successes: float, attempts: float, prior: float = 0.5, strength: float = 8.0) -> float:
    return float((successes + prior * strength) / (attempts + strength))


def _avg(total: float, count: float, default: float, strength: float = 6.0) -> float:
    return float((total + default * strength) / (count + strength))


def _elo_probability(elo_a: float, elo_b: float) -> float:
    return float(1 / (1 + 10 ** ((elo_b - elo_a) / 400)))


def _normal_over_probability(mean: float, line: float, std: float) -> float:
    std = max(float(std), 0.1)
    z = (float(line) - float(mean)) / std
    return float(max(0.02, min(0.98, 0.5 * math.erfc(z / math.sqrt(2)))))


@dataclass
class PlayerRollingStats:
    matches: int = 0
    wins: int = 0
    points_for_sum: float = 0.0
    points_against_sum: float = 0.0
    total_points_sum: float = 0.0
    first_set_total_sum: float = 0.0
    first_set_wins: int = 0
    first_set_over_18_5: int = 0
    sets_played_sum: float = 0.0
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_total_points: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_first_set_total: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_first_over: deque = field(default_factory=lambda: deque(maxlen=20))

    def as_features(self, defaults: dict[str, float]) -> dict[str, float]:
        matches = float(self.matches)
        win_rate = _rate(self.wins, matches, prior=0.5, strength=10)
        recent_win_rate = (
            float(np.mean(self.recent_wins)) if self.recent_wins else win_rate
        )
        avg_total_points = _avg(
            self.total_points_sum, matches, defaults["avg_total_points"], strength=8
        )
        avg_first_set_total = _avg(
            self.first_set_total_sum, matches, defaults["avg_first_set_total"], strength=8
        )
        avg_points_for = _avg(
            self.points_for_sum, matches, defaults["avg_points_for"], strength=8
        )
        avg_points_against = _avg(
            self.points_against_sum,
            matches,
            defaults["avg_points_against"],
            strength=8,
        )
        recent_avg_total = (
            float(np.mean(self.recent_total_points))
            if self.recent_total_points
            else avg_total_points
        )
        recent_avg_first = (
            float(np.mean(self.recent_first_set_total))
            if self.recent_first_set_total
            else avg_first_set_total
        )
        first_over_rate = _rate(
            self.first_set_over_18_5,
            matches,
            prior=defaults["first_set_over_18_5_rate"],
            strength=10,
        )
        recent_first_over_rate = (
            float(np.mean(self.recent_first_over))
            if self.recent_first_over
            else first_over_rate
        )
        first_set_win_rate = _rate(self.first_set_wins, matches, prior=0.5, strength=10)
        return {
            "matches": matches,
            "win_rate": win_rate,
            "recent_win_rate": recent_win_rate,
            "first_set_win_rate": first_set_win_rate,
            "point_diff_avg": avg_points_for - avg_points_against,
            "avg_total_points": avg_total_points,
            "recent_avg_total_points": recent_avg_total,
            "avg_first_set_total": avg_first_set_total,
            "recent_avg_first_set_total": recent_avg_first,
            "first_over_rate": first_over_rate,
            "recent_first_over_rate": recent_first_over_rate,
        }

    def update(
        self,
        won: bool,
        points_for: float,
        points_against: float,
        total_points: float,
        first_set_total: float,
        first_set_won: bool,
        first_set_over_18_5: bool,
        sets_played: int,
    ) -> None:
        self.matches += 1
        self.wins += int(bool(won))
        self.points_for_sum += float(points_for)
        self.points_against_sum += float(points_against)
        self.total_points_sum += float(total_points)
        self.first_set_total_sum += float(first_set_total)
        self.first_set_wins += int(bool(first_set_won))
        self.first_set_over_18_5 += int(bool(first_set_over_18_5))
        self.sets_played_sum += float(sets_played)
        self.recent_wins.append(int(bool(won)))
        self.recent_total_points.append(float(total_points))
        self.recent_first_set_total.append(float(first_set_total))
        self.recent_first_over.append(int(bool(first_set_over_18_5)))


@dataclass
class H2HRollingStats:
    matches: int = 0
    wins_by_player: dict[str, int] = field(default_factory=dict)
    total_points_sum: float = 0.0
    first_set_total_sum: float = 0.0
    first_set_over_18_5: int = 0

    def as_features(self, player_a: str, defaults: dict[str, float]) -> dict[str, float]:
        if self.matches <= 0:
            return {
                "h2h_matches": 0.0,
                "h2h_a_win_rate": 0.5,
                "h2h_a_win_diff": 0.0,
                "h2h_avg_total_points": defaults["avg_total_points"],
                "h2h_avg_first_set_total": defaults["avg_first_set_total"],
                "h2h_first_over_rate": defaults["first_set_over_18_5_rate"],
            }
        a_wins = float(self.wins_by_player.get(player_a, 0))
        win_rate = _rate(a_wins, float(self.matches), prior=0.5, strength=4)
        return {
            "h2h_matches": float(self.matches),
            "h2h_a_win_rate": win_rate,
            "h2h_a_win_diff": win_rate - 0.5,
            "h2h_avg_total_points": float(self.total_points_sum / self.matches),
            "h2h_avg_first_set_total": float(self.first_set_total_sum / self.matches),
            "h2h_first_over_rate": float(self.first_set_over_18_5 / self.matches),
        }

    def update(
        self,
        winner: str,
        total_points: float,
        first_set_total: float,
        first_set_over_18_5: bool,
    ) -> None:
        self.matches += 1
        self.wins_by_player[winner] = self.wins_by_player.get(winner, 0) + 1
        self.total_points_sum += float(total_points)
        self.first_set_total_sum += float(first_set_total)
        self.first_set_over_18_5 += int(bool(first_set_over_18_5))


@dataclass
class RollingFeatureState:
    player_stats: dict[str, PlayerRollingStats] = field(default_factory=dict)
    h2h_stats: dict[tuple[str, str], H2HRollingStats] = field(default_factory=dict)
    elo: dict[str, float] = field(default_factory=dict)
    defaults: dict[str, float] = field(default_factory=dict)
    last_updated: Any = None

    def player(self, name: str) -> PlayerRollingStats:
        if name not in self.player_stats:
            self.player_stats[name] = PlayerRollingStats()
        return self.player_stats[name]

    def h2h(self, player_a: str, player_b: str) -> H2HRollingStats:
        key = tuple(sorted((player_a, player_b)))
        if key not in self.h2h_stats:
            self.h2h_stats[key] = H2HRollingStats()
        return self.h2h_stats[key]

    def elo_for(self, player: str) -> float:
        if player not in self.elo:
            self.elo[player] = 1500.0
        return float(self.elo[player])


# -----------------------------
# Dataset builder
# -----------------------------


def global_defaults(matches: pd.DataFrame) -> dict[str, float]:
    return {
        "avg_total_points": float(matches["total_points"].mean()),
        "std_total_points": float(matches["total_points"].std()),
        "avg_first_set_total": float(matches["first_set_total"].mean()),
        "std_first_set_total": float(matches["first_set_total"].std()),
        "first_set_over_18_5_rate": float(matches["first_set_over_18_5"].mean()),
        "avg_points_for": float(matches[["p1_points", "p2_points"]].stack().mean()),
        "avg_points_against": float(matches[["p1_points", "p2_points"]].stack().mean()),
    }


def make_feature_row(
    state: RollingFeatureState,
    player_a: str,
    player_b: str,
) -> dict[str, float]:
    defaults = state.defaults
    a_stats = state.player(player_a).as_features(defaults)
    b_stats = state.player(player_b).as_features(defaults)
    a_elo = state.elo_for(player_a)
    b_elo = state.elo_for(player_b)
    elo_prob = _elo_probability(a_elo, b_elo)
    h2h_features = state.h2h(player_a, player_b).as_features(player_a, defaults)

    features = {
        "a_elo": a_elo,
        "b_elo": b_elo,
        "elo_diff": a_elo - b_elo,
        "elo_probability": elo_prob,
        "a_matches": a_stats["matches"],
        "b_matches": b_stats["matches"],
        "match_count_diff": a_stats["matches"] - b_stats["matches"],
        "a_win_rate": a_stats["win_rate"],
        "b_win_rate": b_stats["win_rate"],
        "win_rate_diff": a_stats["win_rate"] - b_stats["win_rate"],
        "a_recent_win_rate": a_stats["recent_win_rate"],
        "b_recent_win_rate": b_stats["recent_win_rate"],
        "recent_win_rate_diff": a_stats["recent_win_rate"] - b_stats["recent_win_rate"],
        "a_first_set_win_rate": a_stats["first_set_win_rate"],
        "b_first_set_win_rate": b_stats["first_set_win_rate"],
        "first_set_win_rate_diff": a_stats["first_set_win_rate"] - b_stats["first_set_win_rate"],
        "a_point_diff_avg": a_stats["point_diff_avg"],
        "b_point_diff_avg": b_stats["point_diff_avg"],
        "point_diff_diff": a_stats["point_diff_avg"] - b_stats["point_diff_avg"],
        "a_avg_total_points": a_stats["avg_total_points"],
        "b_avg_total_points": b_stats["avg_total_points"],
        "avg_total_points_mean": (a_stats["avg_total_points"] + b_stats["avg_total_points"]) / 2,
        "avg_total_points_diff": a_stats["avg_total_points"] - b_stats["avg_total_points"],
        "a_recent_avg_total_points": a_stats["recent_avg_total_points"],
        "b_recent_avg_total_points": b_stats["recent_avg_total_points"],
        "recent_avg_total_points_mean": (
            a_stats["recent_avg_total_points"] + b_stats["recent_avg_total_points"]
        )
        / 2,
        "a_avg_first_set_total": a_stats["avg_first_set_total"],
        "b_avg_first_set_total": b_stats["avg_first_set_total"],
        "avg_first_set_total_mean": (
            a_stats["avg_first_set_total"] + b_stats["avg_first_set_total"]
        )
        / 2,
        "a_recent_avg_first_set_total": a_stats["recent_avg_first_set_total"],
        "b_recent_avg_first_set_total": b_stats["recent_avg_first_set_total"],
        "recent_avg_first_set_total_mean": (
            a_stats["recent_avg_first_set_total"]
            + b_stats["recent_avg_first_set_total"]
        )
        / 2,
        "a_first_over_rate": a_stats["first_over_rate"],
        "b_first_over_rate": b_stats["first_over_rate"],
        "first_over_rate_mean": (a_stats["first_over_rate"] + b_stats["first_over_rate"])
        / 2,
        "first_over_rate_diff": a_stats["first_over_rate"] - b_stats["first_over_rate"],
        "a_recent_first_over_rate": a_stats["recent_first_over_rate"],
        "b_recent_first_over_rate": b_stats["recent_first_over_rate"],
        "recent_first_over_rate_mean": (
            a_stats["recent_first_over_rate"] + b_stats["recent_first_over_rate"]
        )
        / 2,
        **h2h_features,
    }
    return {col: float(features[col]) for col in FEATURE_COLUMNS}


def _append_training_rows(
    rows: list[dict[str, Any]],
    state: RollingFeatureState,
    player1: str,
    player2: str,
    winner: str,
    total_points: float,
    first_set_total: float,
    first_set_over_18_5: bool,
    date_time: Any,
    source_match_id: Any,
) -> None:
    p1_features = make_feature_row(state, player1, player2)
    p1_features.update(
        {
            "player_a": player1,
            "player_b": player2,
            "date_time": date_time,
            "source_match_id": source_match_id,
            "orientation": "p1_vs_p2",
            TARGET_WIN: int(winner == player1),
            TARGET_FIRST_OVER: int(bool(first_set_over_18_5)),
            TARGET_TOTAL_POINTS: float(total_points),
            TARGET_FIRST_SET_POINTS: float(first_set_total),
        }
    )
    rows.append(p1_features)

    p2_features = make_feature_row(state, player2, player1)
    p2_features.update(
        {
            "player_a": player2,
            "player_b": player1,
            "date_time": date_time,
            "source_match_id": source_match_id,
            "orientation": "p2_vs_p1",
            TARGET_WIN: int(winner == player2),
            TARGET_FIRST_OVER: int(bool(first_set_over_18_5)),
            TARGET_TOTAL_POINTS: float(total_points),
            TARGET_FIRST_SET_POINTS: float(first_set_total),
        }
    )
    rows.append(p2_features)


def _update_state_from_match(state: RollingFeatureState, row: Any, k_factor: float) -> None:
    p1 = str(row.player1)
    p2 = str(row.player2)
    winner = str(row.winner)
    p1_won = bool(row.p1_won)

    p1_elo = state.elo_for(p1)
    p2_elo = state.elo_for(p2)
    p1_expected = _elo_probability(p1_elo, p2_elo)
    p1_score = 1.0 if p1_won else 0.0
    state.elo[p1] = p1_elo + k_factor * (p1_score - p1_expected)
    state.elo[p2] = p2_elo + k_factor * ((1.0 - p1_score) - (1.0 - p1_expected))

    total_points = float(row.total_points)
    first_set_total = float(row.first_set_total)
    first_over = bool(row.first_set_over_18_5)

    state.player(p1).update(
        won=p1_won,
        points_for=float(row.p1_points),
        points_against=float(row.p2_points),
        total_points=total_points,
        first_set_total=first_set_total,
        first_set_won=bool(row.first_set_p1_won),
        first_set_over_18_5=first_over,
        sets_played=int(row.sets_played),
    )
    state.player(p2).update(
        won=not p1_won,
        points_for=float(row.p2_points),
        points_against=float(row.p1_points),
        total_points=total_points,
        first_set_total=first_set_total,
        first_set_won=bool(row.first_set_p2_won),
        first_set_over_18_5=first_over,
        sets_played=int(row.sets_played),
    )
    state.h2h(p1, p2).update(
        winner=winner,
        total_points=total_points,
        first_set_total=first_set_total,
        first_set_over_18_5=first_over,
    )
    state.last_updated = row.date_time


def build_feature_frame(
    matches: pd.DataFrame,
    k_factor: float = 24.0,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, RollingFeatureState]:
    """Create chronological pre-match ML features and the final rolling state.

    Two training rows are created per match:
    - player1 vs player2
    - player2 vs player1

    This teaches the model to predict "Player A wins" without learning a fake
    player1/player2 side advantage.

    If max_rows is provided, only the most recent orientation rows are retained
    for fitting while the rolling state still processes every match.
    """
    df = matches.copy()
    if "total_points" not in df.columns:
        df = enrich_matches(df)
    df = df.sort_values(["date_time", "source_match_id"]).reset_index(drop=True)

    state = RollingFeatureState(defaults=global_defaults(df))
    rows = deque(maxlen=int(max_rows)) if max_rows and max_rows > 0 else []
    start_feature_match = 0
    if max_rows and max_rows > 0:
        # Each match creates two orientation rows. For capped training, skip
        # feature-row creation until the recent window while still updating
        # rolling state for the full historical context.
        start_feature_match = max(0, len(df) - int(math.ceil(max_rows / 2)))

    for idx, row in enumerate(df.itertuples(index=False)):
        if idx >= start_feature_match:
            _append_training_rows(
                rows=rows,
                state=state,
                player1=str(row.player1),
                player2=str(row.player2),
                winner=str(row.winner),
                total_points=float(row.total_points),
                first_set_total=float(row.first_set_total),
                first_set_over_18_5=bool(row.first_set_over_18_5),
                date_time=row.date_time,
                source_match_id=row.source_match_id,
            )
        _update_state_from_match(state, row, k_factor=k_factor)

    features = pd.DataFrame(rows)
    return features, state


# -----------------------------
# Training and evaluation
# -----------------------------


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _make_classifier(algorithm: str, random_state: int):
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=280,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=random_state,
    )


def _make_regressor(algorithm: str, random_state: int):
    if algorithm == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=random_state,
    )


def _predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    # Fallback for unusual estimators.
    pred = model.predict(x)
    return np.clip(pred.astype(float), 0.0, 1.0)


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "residual_std": float(np.std(y_true - y_pred)),
    }


def _select_algorithm(algorithm: str) -> str:
    if algorithm == "auto":
        return "xgboost" if _xgboost_available() else "sklearn"
    if algorithm not in {"xgboost", "sklearn"}:
        raise ValueError("algorithm must be 'auto', 'xgboost', or 'sklearn'")
    if algorithm == "xgboost" and not _xgboost_available():
        raise ImportError("xgboost is not installed. Install requirements or use algorithm='sklearn'.")
    return algorithm


def train_model_bundle(
    matches: pd.DataFrame,
    algorithm: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    max_training_rows: int | None = None,
) -> dict[str, Any]:
    """Train winner, first-set O18.5, total-points, and first-set-points models."""
    selected_algorithm = _select_algorithm(algorithm)
    available_rows = int(len(matches) * 2)
    feature_frame, state = build_feature_frame(matches, max_rows=max_training_rows)
    feature_frame = feature_frame.sort_values(["date_time", "source_match_id", "orientation"])
    training_frame = feature_frame.copy()

    split_idx = max(1, int(len(training_frame) * (1 - test_size)))
    if split_idx >= len(training_frame):
        split_idx = len(training_frame) - 1

    train_df = training_frame.iloc[:split_idx].copy()
    test_df = training_frame.iloc[split_idx:].copy()
    x_train = train_df[FEATURE_COLUMNS].astype(float)
    x_test = test_df[FEATURE_COLUMNS].astype(float)

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    target_map = {
        "winner": TARGET_WIN,
        "first_set_over_18_5": TARGET_FIRST_OVER,
    }
    for name, target in target_map.items():
        model = _make_classifier(selected_algorithm, random_state=random_state)
        y_train = train_df[target].astype(int).to_numpy()
        y_test = test_df[target].astype(int).to_numpy()
        model.fit(x_train, y_train)
        y_prob = _predict_probability(model, x_test)
        models[name] = model
        metrics[name] = _classification_metrics(y_test, y_prob)

    regression_target_map = {
        "total_points": TARGET_TOTAL_POINTS,
        "first_set_points": TARGET_FIRST_SET_POINTS,
    }
    for name, target in regression_target_map.items():
        model = _make_regressor(selected_algorithm, random_state=random_state)
        y_train = train_df[target].astype(float).to_numpy()
        y_test = test_df[target].astype(float).to_numpy()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        models[name] = model
        metrics[name] = _regression_metrics(y_test, y_pred)

    bundle = {
        "models": models,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "feature_state": state,
        "algorithm": selected_algorithm,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_total": available_rows,
        "rows_used_for_training": int(len(training_frame)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "date_min": str(feature_frame["date_time"].min()),
        "date_max": str(feature_frame["date_time"].max()),
    }
    return bundle


# -----------------------------
# Prediction and persistence
# -----------------------------


def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_with_bundle(
    bundle: dict[str, Any],
    player_a: str,
    player_b: str,
    first_set_line: float = 18.5,
    total_points_line: float = 75.5,
) -> dict[str, Any]:
    if player_a == player_b:
        raise ValueError("Choose two different players.")

    state: RollingFeatureState = bundle["feature_state"]
    feature_row = make_feature_row(state, player_a, player_b)
    x = pd.DataFrame([feature_row], columns=bundle["feature_columns"]).astype(float)
    models = bundle["models"]

    winner_probability = float(_predict_probability(models["winner"], x)[0])
    first_o18_probability = float(_predict_probability(models["first_set_over_18_5"], x)[0])
    expected_total_points = float(models["total_points"].predict(x)[0])
    expected_first_set_points = float(models["first_set_points"].predict(x)[0])

    total_std = float(
        bundle["metrics"].get("total_points", {}).get(
            "residual_std", state.defaults.get("std_total_points", 16.6)
        )
    )
    first_std = float(
        bundle["metrics"].get("first_set_points", {}).get(
            "residual_std", state.defaults.get("std_first_set_total", 3.2)
        )
    )

    total_over_probability = _normal_over_probability(
        expected_total_points, total_points_line, total_std
    )
    first_line_model_probability = _normal_over_probability(
        expected_first_set_points, first_set_line, first_std
    )
    if abs(first_set_line - 18.5) < 1e-9:
        first_over_probability = float(
            max(0.02, min(0.98, 0.65 * first_o18_probability + 0.35 * first_line_model_probability))
        )
    else:
        first_over_probability = first_line_model_probability

    return {
        "player_a": player_a,
        "player_b": player_b,
        "player_a_win_probability": winner_probability,
        "player_b_win_probability": 1 - winner_probability,
        "predicted_winner": player_a if winner_probability >= 0.5 else player_b,
        "expected_total_points": expected_total_points,
        "total_points_line": float(total_points_line),
        "total_points_over_probability": total_over_probability,
        "total_points_under_probability": 1 - total_over_probability,
        "expected_first_set_points": expected_first_set_points,
        "first_set_line": float(first_set_line),
        "first_set_over_probability": first_over_probability,
        "first_set_under_probability": 1 - first_over_probability,
        "raw_first_set_o18_5_probability": first_o18_probability,
        "algorithm": bundle["algorithm"],
        "trained_at_utc": bundle["trained_at_utc"],
        "feature_row": feature_row,
    }


def metrics_table(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, model_metrics in bundle.get("metrics", {}).items():
        row = {"model": model_name}
        row.update(model_metrics)
        rows.append(row)
    return pd.DataFrame(rows)
