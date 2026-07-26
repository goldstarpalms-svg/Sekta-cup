from __future__ import annotations

import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .setka_core import enrich_matches

FEATURE_COLUMNS = [
    "a_elo", "b_elo", "elo_diff", "elo_probability",
    "a_elo_momentum", "b_elo_momentum", "elo_momentum_diff",
    "a_matches", "b_matches", "match_count_diff",
    "a_win_rate", "b_win_rate", "win_rate_diff",
    "a_recent_win_rate", "b_recent_win_rate", "recent_win_rate_diff",
    "a_weighted_recent_form", "b_weighted_recent_form", "weighted_form_diff",
    "a_set_win_rate", "b_set_win_rate", "set_win_rate_diff",
    "a_first_set_win_rate", "b_first_set_win_rate", "first_set_win_rate_diff",
    "a_point_diff_avg", "b_point_diff_avg", "point_diff_diff",
    "a_recent_point_diff", "b_recent_point_diff", "recent_point_diff_diff",
    "a_avg_total_points", "b_avg_total_points",
    "avg_total_points_mean", "avg_total_points_diff",
    "a_recent_avg_total_points", "b_recent_avg_total_points",
    "recent_avg_total_points_mean",
    "a_avg_first_set_total", "b_avg_first_set_total",
    "avg_first_set_total_mean",
    "a_recent_avg_first_set_total", "b_recent_avg_first_set_total",
    "recent_avg_first_set_total_mean",
    "a_first_over_rate", "b_first_over_rate",
    "first_over_rate_mean", "first_over_rate_diff",
    "a_recent_first_over_rate", "b_recent_first_over_rate",
    "recent_first_over_rate_mean",
    "a_hours_since_last", "b_hours_since_last",
    "a_matches_today", "b_matches_today", "matches_today_diff",
    "a_days_since_last_win", "b_days_since_last_win",
    "a_win_streak", "b_win_streak",
    "a_loss_streak", "b_loss_streak", "streak_diff",
    "h2h_matches", "h2h_a_win_rate", "h2h_a_win_diff",
    "h2h_avg_total_points", "h2h_avg_first_set_total",
    "h2h_first_over_rate", "h2h_recent_a_win_rate",
]

TARGET_WIN = "target_a_win"
TARGET_FIRST_OVER = "target_first_set_over_18_5"
TARGET_TOTAL_POINTS = "target_total_points"
TARGET_FIRST_SET_POINTS = "target_first_set_points"


def _rate(successes, attempts, prior=0.5, strength=8.0):
    return float((successes + prior * strength) / (attempts + strength))


def _avg(total, count, default, strength=6.0):
    return float((total + default * strength) / (count + strength))


def _elo_probability(elo_a, elo_b):
    return float(1 / (1 + 10 ** ((elo_b - elo_a) / 400)))


def _normal_over_probability(mean, line, std):
    std = max(float(std), 0.1)
    z = (float(line) - float(mean)) / std
    return float(max(0.02, min(0.98, 0.5 * math.erfc(z / math.sqrt(2)))))


def _dynamic_k_factor(match_count):
    if match_count < 30:
        return 40.0
    elif match_count < 100:
        return 32.0
    else:
        return 24.0


def _margin_multiplier(sets_won, sets_lost):
    margin = abs(sets_won - sets_lost)
    return 1.0 + (margin - 1) * 0.15



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
    sets_won_total: int = 0
    sets_lost_total: int = 0
    sets_played_sum: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    last_match_datetime: Any = None
    last_win_datetime: Any = None
    elo_history: deque = field(default_factory=lambda: deque(maxlen=10))
    match_datetimes_today: list = field(default_factory=list)
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_total_points: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_first_set_total: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_first_over: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_point_diffs: deque = field(default_factory=lambda: deque(maxlen=15))

    def hours_since_last_match(self, current_dt):
        if self.last_match_datetime is None or pd.isna(current_dt):
            return 168.0
        try:
            delta = pd.Timestamp(current_dt) - pd.Timestamp(self.last_match_datetime)
            hours = delta.total_seconds() / 3600.0
            return float(max(0.0, min(hours, 720.0)))
        except Exception:
            return 168.0

    def matches_today(self, current_dt):
        if pd.isna(current_dt):
            return 0
        try:
            current_date = pd.Timestamp(current_dt).date()
            return sum(1 for dt in self.match_datetimes_today
                       if pd.Timestamp(dt).date() == current_date)
        except Exception:
            return 0

    def days_since_last_win(self, current_dt):
        if self.last_win_datetime is None or pd.isna(current_dt):
            return 30.0
        try:
            delta = pd.Timestamp(current_dt) - pd.Timestamp(self.last_win_datetime)
            days = delta.total_seconds() / 86400.0
            return float(max(0.0, min(days, 90.0)))
        except Exception:
            return 30.0

    def weighted_recent_form(self):
        if not self.recent_wins:
            return 0.5
        wins = list(self.recent_wins)
        n = len(wins)
        weights = np.exp(-0.08 * np.arange(n - 1, -1, -1))
        return float(np.average(wins, weights=weights))

    def elo_momentum(self):
        if len(self.elo_history) < 2:
            return 0.0
        return float(self.elo_history[-1] - self.elo_history[0])

    def as_features(self, defaults, current_dt=None):
        matches = float(self.matches)
        win_rate = _rate(self.wins, matches, prior=0.5, strength=10)
        recent_win_rate = float(np.mean(self.recent_wins)) if self.recent_wins else win_rate
        weighted_form = self.weighted_recent_form()
        total_sets = self.sets_won_total + self.sets_lost_total
        set_win_rate = _rate(self.sets_won_total, total_sets, prior=0.5, strength=15) if total_sets > 0 else 0.5
        avg_total_points = _avg(self.total_points_sum, matches, defaults["avg_total_points"], strength=8)
        avg_first_set_total = _avg(self.first_set_total_sum, matches, defaults["avg_first_set_total"], strength=8)
        avg_points_for = _avg(self.points_for_sum, matches, defaults["avg_points_for"], strength=8)
        avg_points_against = _avg(self.points_against_sum, matches, defaults["avg_points_against"], strength=8)
        recent_avg_total = float(np.mean(self.recent_total_points)) if self.recent_total_points else avg_total_points
        recent_avg_first = float(np.mean(self.recent_first_set_total)) if self.recent_first_set_total else avg_first_set_total
        first_over_rate = _rate(self.first_set_over_18_5, matches, prior=defaults["first_set_over_18_5_rate"], strength=10)
        recent_first_over_rate = float(np.mean(self.recent_first_over)) if self.recent_first_over else first_over_rate
        first_set_win_rate = _rate(self.first_set_wins, matches, prior=0.5, strength=10)
        recent_point_diff = float(np.mean(self.recent_point_diffs)) if self.recent_point_diffs else avg_points_for - avg_points_against

        return {
            "matches": matches,
            "win_rate": win_rate,
            "recent_win_rate": recent_win_rate,
            "weighted_recent_form": weighted_form,
            "set_win_rate": set_win_rate,
            "first_set_win_rate": first_set_win_rate,
            "point_diff_avg": avg_points_for - avg_points_against,
            "recent_point_diff": recent_point_diff,
            "avg_total_points": avg_total_points,
            "recent_avg_total_points": recent_avg_total,
            "avg_first_set_total": avg_first_set_total,
            "recent_avg_first_set_total": recent_avg_first,
            "first_over_rate": first_over_rate,
            "recent_first_over_rate": recent_first_over_rate,
            "hours_since_last": self.hours_since_last_match(current_dt),
            "matches_today": float(self.matches_today(current_dt)),
            "days_since_last_win": self.days_since_last_win(current_dt),
            "win_streak": float(self.current_win_streak),
            "loss_streak": float(self.current_loss_streak),
            "elo_momentum": self.elo_momentum(),
        }

    def update(self, won, points_for, points_against, total_points, first_set_total,
               first_set_won, first_set_over_18_5, sets_played, sets_won, sets_lost,
               match_datetime, new_elo):
        self.matches += 1
        self.wins += int(bool(won))
        self.points_for_sum += float(points_for)
        self.points_against_sum += float(points_against)
        self.total_points_sum += float(total_points)
        self.first_set_total_sum += float(first_set_total)
        self.first_set_wins += int(bool(first_set_won))
        self.first_set_over_18_5 += int(bool(first_set_over_18_5))
        self.sets_played_sum += float(sets_played)
        self.sets_won_total += int(sets_won)
        self.sets_lost_total += int(sets_lost)
        if won:
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.last_win_datetime = match_datetime
        else:
            self.current_loss_streak += 1
            self.current_win_streak = 0
        self.last_match_datetime = match_datetime
        self.match_datetimes_today.append(match_datetime)
        if len(self.match_datetimes_today) > 50:
            self.match_datetimes_today = self.match_datetimes_today[-50:]
        self.elo_history.append(float(new_elo))
        point_diff = float(points_for) - float(points_against)
        self.recent_point_diffs.append(point_diff)
        self.recent_wins.append(int(bool(won)))
        self.recent_total_points.append(float(total_points))
        self.recent_first_set_total.append(float(first_set_total))
        self.recent_first_over.append(int(bool(first_set_over_18_5)))


@dataclass
class H2HRollingStats:
    matches: int = 0
    wins_by_player: dict = field(default_factory=dict)
    total_points_sum: float = 0.0
    first_set_total_sum: float = 0.0
    first_set_over_18_5: int = 0
    recent_results: deque = field(default_factory=lambda: deque(maxlen=5))

    def as_features(self, player_a, defaults):
        if self.matches <= 0:
            return {
                "h2h_matches": 0.0, "h2h_a_win_rate": 0.5, "h2h_a_win_diff": 0.0,
                "h2h_avg_total_points": defaults["avg_total_points"],
                "h2h_avg_first_set_total": defaults["avg_first_set_total"],
                "h2h_first_over_rate": defaults["first_set_over_18_5_rate"],
                "h2h_recent_a_win_rate": 0.5,
            }
        a_wins = float(self.wins_by_player.get(player_a, 0))
        win_rate = _rate(a_wins, float(self.matches), prior=0.5, strength=4)
        recent_a_wins = sum(1 for winner in self.recent_results if winner == player_a)
        recent_h2h_rate = recent_a_wins / len(self.recent_results) if self.recent_results else 0.5
        return {
            "h2h_matches": float(self.matches),
            "h2h_a_win_rate": win_rate,
            "h2h_a_win_diff": win_rate - 0.5,
            "h2h_avg_total_points": float(self.total_points_sum / self.matches),
            "h2h_avg_first_set_total": float(self.first_set_total_sum / self.matches),
            "h2h_first_over_rate": float(self.first_set_over_18_5 / self.matches),
            "h2h_recent_a_win_rate": float(recent_h2h_rate),
        }

    def update(self, winner, total_points, first_set_total, first_set_over_18_5):
        self.matches += 1
        self.wins_by_player[winner] = self.wins_by_player.get(winner, 0) + 1
        self.total_points_sum += float(total_points)
        self.first_set_total_sum += float(first_set_total)
        self.first_set_over_18_5 += int(bool(first_set_over_18_5))
        self.recent_results.append(winner)



@dataclass
class RollingFeatureState:
    player_stats: dict = field(default_factory=dict)
    h2h_stats: dict = field(default_factory=dict)
    elo: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    last_updated: Any = None

    def player(self, name):
        if name not in self.player_stats:
            self.player_stats[name] = PlayerRollingStats()
        return self.player_stats[name]

    def h2h(self, player_a, player_b):
        key = tuple(sorted((player_a, player_b)))
        if key not in self.h2h_stats:
            self.h2h_stats[key] = H2HRollingStats()
        return self.h2h_stats[key]

    def elo_for(self, player):
        if player not in self.elo:
            self.elo[player] = 1500.0
        return float(self.elo[player])


def global_defaults(matches):
    return {
        "avg_total_points": float(matches["total_points"].mean()),
        "std_total_points": float(matches["total_points"].std()),
        "avg_first_set_total": float(matches["first_set_total"].mean()),
        "std_first_set_total": float(matches["first_set_total"].std()),
        "first_set_over_18_5_rate": float(matches["first_set_over_18_5"].mean()),
        "avg_points_for": float(matches[["p1_points", "p2_points"]].stack().mean()),
        "avg_points_against": float(matches[["p1_points", "p2_points"]].stack().mean()),
    }


def make_feature_row(state, player_a, player_b, current_dt=None):
    defaults = state.defaults
    a_stats = state.player(player_a).as_features(defaults, current_dt)
    b_stats = state.player(player_b).as_features(defaults, current_dt)
    a_elo = state.elo_for(player_a)
    b_elo = state.elo_for(player_b)
    elo_prob = _elo_probability(a_elo, b_elo)
    h2h_features = state.h2h(player_a, player_b).as_features(player_a, defaults)

    features = {
        "a_elo": a_elo, "b_elo": b_elo,
        "elo_diff": a_elo - b_elo, "elo_probability": elo_prob,
        "a_elo_momentum": a_stats["elo_momentum"],
        "b_elo_momentum": b_stats["elo_momentum"],
        "elo_momentum_diff": a_stats["elo_momentum"] - b_stats["elo_momentum"],
        "a_matches": a_stats["matches"], "b_matches": b_stats["matches"],
        "match_count_diff": a_stats["matches"] - b_stats["matches"],
        "a_win_rate": a_stats["win_rate"], "b_win_rate": b_stats["win_rate"],
        "win_rate_diff": a_stats["win_rate"] - b_stats["win_rate"],
        "a_recent_win_rate": a_stats["recent_win_rate"],
        "b_recent_win_rate": b_stats["recent_win_rate"],
        "recent_win_rate_diff": a_stats["recent_win_rate"] - b_stats["recent_win_rate"],
        "a_weighted_recent_form": a_stats["weighted_recent_form"],
        "b_weighted_recent_form": b_stats["weighted_recent_form"],
        "weighted_form_diff": a_stats["weighted_recent_form"] - b_stats["weighted_recent_form"],
        "a_set_win_rate": a_stats["set_win_rate"],
        "b_set_win_rate": b_stats["set_win_rate"],
        "set_win_rate_diff": a_stats["set_win_rate"] - b_stats["set_win_rate"],
        "a_first_set_win_rate": a_stats["first_set_win_rate"],
        "b_first_set_win_rate": b_stats["first_set_win_rate"],
        "first_set_win_rate_diff": a_stats["first_set_win_rate"] - b_stats["first_set_win_rate"],
        "a_point_diff_avg": a_stats["point_diff_avg"],
        "b_point_diff_avg": b_stats["point_diff_avg"],
        "point_diff_diff": a_stats["point_diff_avg"] - b_stats["point_diff_avg"],
        "a_recent_point_diff": a_stats["recent_point_diff"],
        "b_recent_point_diff": b_stats["recent_point_diff"],
        "recent_point_diff_diff": a_stats["recent_point_diff"] - b_stats["recent_point_diff"],
        "a_avg_total_points": a_stats["avg_total_points"],
        "b_avg_total_points": b_stats["avg_total_points"],
        "avg_total_points_mean": (a_stats["avg_total_points"] + b_stats["avg_total_points"]) / 2,
        "avg_total_points_diff": a_stats["avg_total_points"] - b_stats["avg_total_points"],
        "a_recent_avg_total_points": a_stats["recent_avg_total_points"],
        "b_recent_avg_total_points": b_stats["recent_avg_total_points"],
        "recent_avg_total_points_mean": (a_stats["recent_avg_total_points"] + b_stats["recent_avg_total_points"]) / 2,
        "a_avg_first_set_total": a_stats["avg_first_set_total"],
        "b_avg_first_set_total": b_stats["avg_first_set_total"],
        "avg_first_set_total_mean": (a_stats["avg_first_set_total"] + b_stats["avg_first_set_total"]) / 2,
        "a_recent_avg_first_set_total": a_stats["recent_avg_first_set_total"],
        "b_recent_avg_first_set_total": b_stats["recent_avg_first_set_total"],
        "recent_avg_first_set_total_mean": (a_stats["recent_avg_first_set_total"] + b_stats["recent_avg_first_set_total"]) / 2,
        "a_first_over_rate": a_stats["first_over_rate"],
        "b_first_over_rate": b_stats["first_over_rate"],
        "first_over_rate_mean": (a_stats["first_over_rate"] + b_stats["first_over_rate"]) / 2,
        "first_over_rate_diff": a_stats["first_over_rate"] - b_stats["first_over_rate"],
        "a_recent_first_over_rate": a_stats["recent_first_over_rate"],
        "b_recent_first_over_rate": b_stats["recent_first_over_rate"],
        "recent_first_over_rate_mean": (a_stats["recent_first_over_rate"] + b_stats["recent_first_over_rate"]) / 2,
        "a_hours_since_last": a_stats["hours_since_last"],
        "b_hours_since_last": b_stats["hours_since_last"],
        "a_matches_today": a_stats["matches_today"],
        "b_matches_today": b_stats["matches_today"],
        "matches_today_diff": a_stats["matches_today"] - b_stats["matches_today"],
        "a_days_since_last_win": a_stats["days_since_last_win"],
        "b_days_since_last_win": b_stats["days_since_last_win"],
        "a_win_streak": a_stats["win_streak"],
        "b_win_streak": b_stats["win_streak"],
        "a_loss_streak": a_stats["loss_streak"],
        "b_loss_streak": b_stats["loss_streak"],
        "streak_diff": (a_stats["win_streak"] - a_stats["loss_streak"]) - (b_stats["win_streak"] - b_stats["loss_streak"]),
        **h2h_features,
    }
    return {col: float(features[col]) for col in FEATURE_COLUMNS}


def _append_training_rows(rows, state, player1, player2, winner, total_points,
                          first_set_total, first_set_over_18_5, date_time, source_match_id):
    p1_features = make_feature_row(state, player1, player2, date_time)
    p1_features.update({
        "player_a": player1, "player_b": player2,
        "date_time": date_time, "source_match_id": source_match_id,
        "orientation": "p1_vs_p2",
        TARGET_WIN: int(winner == player1),
        TARGET_FIRST_OVER: int(bool(first_set_over_18_5)),
        TARGET_TOTAL_POINTS: float(total_points),
        TARGET_FIRST_SET_POINTS: float(first_set_total),
    })
    rows.append(p1_features)

    p2_features = make_feature_row(state, player2, player1, date_time)
    p2_features.update({
        "player_a": player2, "player_b": player1,
        "date_time": date_time, "source_match_id": source_match_id,
        "orientation": "p2_vs_p1",
        TARGET_WIN: int(winner == player2),
        TARGET_FIRST_OVER: int(bool(first_set_over_18_5)),
        TARGET_TOTAL_POINTS: float(total_points),
        TARGET_FIRST_SET_POINTS: float(first_set_total),
    })
    rows.append(p2_features)


def _update_state_from_match(state, row, k_factor_base=32.0):
    p1 = str(row.player1)
    p2 = str(row.player2)
    winner = str(row.winner)
    p1_won = bool(row.p1_won)
    match_dt = row.date_time

    p1_elo = state.elo_for(p1)
    p2_elo = state.elo_for(p2)
    p1_expected = _elo_probability(p1_elo, p2_elo)
    p1_score = 1.0 if p1_won else 0.0

    p1_matches = state.player(p1).matches
    p2_matches = state.player(p2).matches
    k_p1 = _dynamic_k_factor(p1_matches)
    k_p2 = _dynamic_k_factor(p2_matches)

    p1_sets = int(row.p1_sets_won)
    p2_sets = int(row.p2_sets_won)
    margin_mult = _margin_multiplier(p1_sets, p2_sets)

    new_p1_elo = p1_elo + k_p1 * margin_mult * (p1_score - p1_expected)
    new_p2_elo = p2_elo + k_p2 * margin_mult * ((1.0 - p1_score) - (1.0 - p1_expected))

    state.elo[p1] = new_p1_elo
    state.elo[p2] = new_p2_elo

    total_points = float(row.total_points)
    first_set_total = float(row.first_set_total)
    first_over = bool(row.first_set_over_18_5)

    state.player(p1).update(
        won=p1_won, points_for=float(row.p1_points), points_against=float(row.p2_points),
        total_points=total_points, first_set_total=first_set_total,
        first_set_won=bool(row.first_set_p1_won), first_set_over_18_5=first_over,
        sets_played=int(row.sets_played), sets_won=p1_sets, sets_lost=p2_sets,
        match_datetime=match_dt, new_elo=new_p1_elo,
    )
    state.player(p2).update(
        won=not p1_won, points_for=float(row.p2_points), points_against=float(row.p1_points),
        total_points=total_points, first_set_total=first_set_total,
        first_set_won=bool(row.first_set_p2_won), first_set_over_18_5=first_over,
        sets_played=int(row.sets_played), sets_won=p2_sets, sets_lost=p1_sets,
        match_datetime=match_dt, new_elo=new_p2_elo,
    )
    state.h2h(p1, p2).update(
        winner=winner, total_points=total_points,
        first_set_total=first_set_total, first_set_over_18_5=first_over,
    )
    state.last_updated = match_dt



def build_feature_frame(matches, k_factor=32.0, max_rows=None):
    df = matches.copy()
    if "total_points" not in df.columns:
        df = enrich_matches(df)
    df = df.sort_values(["date_time", "source_match_id"]).reset_index(drop=True)

    state = RollingFeatureState(defaults=global_defaults(df))
    rows = deque(maxlen=int(max_rows)) if max_rows and max_rows > 0 else []
    start_feature_match = 0
    if max_rows and max_rows > 0:
        start_feature_match = max(0, len(df) - int(math.ceil(max_rows / 2)))

    for idx, row in enumerate(df.itertuples(index=False)):
        if idx >= start_feature_match:
            _append_training_rows(
                rows=rows, state=state,
                player1=str(row.player1), player2=str(row.player2),
                winner=str(row.winner), total_points=float(row.total_points),
                first_set_total=float(row.first_set_total),
                first_set_over_18_5=bool(row.first_set_over_18_5),
                date_time=row.date_time, source_match_id=row.source_match_id,
            )
        _update_state_from_match(state, row, k_factor_base=k_factor)

    features = pd.DataFrame(rows)
    return features, state


def _xgboost_available():
    try:
        import xgboost
        return True
    except Exception:
        return False


def _make_classifier(algorithm, random_state):
    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
            random_state=random_state,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=63,
        l2_regularization=0.1, random_state=random_state,
    )


def _make_regressor(algorithm, random_state):
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror",
            tree_method="hist", n_jobs=-1, random_state=random_state,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=63,
        l2_regularization=0.1, random_state=random_state,
    )


def _predict_probability(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return np.clip(pred.astype(float), 0.0, 1.0)


def train_model_bundle(matches, algorithm="xgboost", max_training_rows=None,
                       test_fraction=0.2, random_state=42):
    from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                                  roc_auc_score, mean_absolute_error,
                                  mean_squared_error, r2_score)

    if algorithm == "xgboost" and not _xgboost_available():
        algorithm = "sklearn"

    features_df, state = build_feature_frame(matches, max_rows=max_training_rows)

    if features_df.empty:
        raise ValueError("No features generated - check your data")

    features_df = features_df.sort_values("date_time").reset_index(drop=True)
    split_idx = int(len(features_df) * (1 - test_fraction))
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS].fillna(0.0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0.0)

    models = {}
    metrics = []

    y_train = train_df[TARGET_WIN]
    y_test = test_df[TARGET_WIN]
    winner_model = _make_classifier(algorithm, random_state)
    winner_model.fit(X_train, y_train)
    y_pred_proba = _predict_probability(winner_model, X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics.append({
        "model": "winner",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier": float(brier_score_loss(y_test, y_pred_proba)),
        "log_loss": float(log_loss(y_test, np.clip(y_pred_proba, 0.001, 0.999))),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "mae": None, "rmse": None, "r2": None, "residual_std": None,
    })
    models["winner"] = winner_model

    y_train = train_df[TARGET_FIRST_OVER]
    y_test = test_df[TARGET_FIRST_OVER]
    first_over_model = _make_classifier(algorithm, random_state)
    first_over_model.fit(X_train, y_train)
    y_pred_proba = _predict_probability(first_over_model, X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics.append({
        "model": "first_set_over_18_5",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier": float(brier_score_loss(y_test, y_pred_proba)),
        "log_loss": float(log_loss(y_test, np.clip(y_pred_proba, 0.001, 0.999))),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "mae": None, "rmse": None, "r2": None, "residual_std": None,
    })
    models["first_set_over_18_5"] = first_over_model

    y_train = train_df[TARGET_TOTAL_POINTS]
    y_test = test_df[TARGET_TOTAL_POINTS]
    total_model = _make_regressor(algorithm, random_state)
    total_model.fit(X_train, y_train)
    y_pred = total_model.predict(X_test)
    residuals = y_test - y_pred
    metrics.append({
        "model": "total_points",
        "accuracy": None, "brier": None, "log_loss": None, "roc_auc": None,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "residual_std": float(residuals.std()),
    })
    models["total_points"] = total_model

    y_train = train_df[TARGET_FIRST_SET_POINTS]
    y_test = test_df[TARGET_FIRST_SET_POINTS]
    first_pts_model = _make_regressor(algorithm, random_state)
    first_pts_model.fit(X_train, y_train)
    y_pred = first_pts_model.predict(X_test)
    residuals = y_test - y_pred
    metrics.append({
        "model": "first_set_points",
        "accuracy": None, "brier": None, "log_loss": None, "roc_auc": None,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "residual_std": float(residuals.std()),
    })
    models["first_set_points"] = first_pts_model

    bundle = {
        "models": models,
        "state": state,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "algorithm": algorithm,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "total_rows": len(features_df),
        "rows_used": len(features_df),
        "n_rows": len(features_df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_models": len(models),
        "n_features": len(FEATURE_COLUMNS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return bundle


def metrics_table(bundle):
    return pd.DataFrame(bundle.get("metrics", []))


def predict_with_bundle(bundle, player_a, player_b, current_dt=None,
                        first_set_line=18.5, total_points_line=75.5, sets_line=3.5):
    """Enhanced ML prediction returning a dict compatible with the app's UI layer.

    Returns fields matching the rule-based predict_match() so prediction_pick_row
    can swap in ML predictions transparently.  Sets over/under is a fallback
    (0.5) because the ML pipeline does not model set count directly.
    """
    state = bundle["state"]
    models = bundle["models"]
    if current_dt is None:
        current_dt = pd.Timestamp.now()
    features = make_feature_row(state, player_a, player_b, current_dt)
    X = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0.0)
    winner_prob = float(_predict_probability(models["winner"], X)[0])
    first_over_18_5_prob = float(_predict_probability(models["first_set_over_18_5"], X)[0])
    total_pred = float(models["total_points"].predict(X)[0])
    first_pts_pred = float(models["first_set_points"].predict(X)[0])

    player_a_win_probability = winner_prob
    player_b_win_probability = 1.0 - winner_prob

    # --- confidence score (0-100 scale, higher = better) ---
    max_winner_prob = max(winner_prob, player_b_win_probability)
    if max_winner_prob >= 0.65:
        confidence = "High"
    elif max_winner_prob >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"
    confidence_score = max_winner_prob * 100.0

    # --- calibration residuals from holdout metrics ---
    total_residual_std = None
    first_residual_std = None
    for m in bundle.get("metrics", []):
        if m["model"] == "total_points":
            total_residual_std = m.get("residual_std")
        elif m["model"] == "first_set_points":
            first_residual_std = m.get("residual_std")

    # --- total points over/under via normal CDF ---
    if total_residual_std and total_residual_std > 0.1:
        total_over_prob = _normal_over_probability(total_pred, total_points_line, total_residual_std)
    else:
        total_over_prob = 0.5
    total_under_prob = 1.0 - total_over_prob

    # --- first-set over/under ---
    if abs(first_set_line - 18.5) < 0.01:
        first_over_prob = first_over_18_5_prob
    elif first_residual_std and first_residual_std > 0.1:
        first_over_prob = _normal_over_probability(first_pts_pred, first_set_line, first_residual_std)
    else:
        first_over_prob = 0.5
    first_under_prob = 1.0 - first_over_prob

    # --- Elo ---
    a_elo = state.elo_for(player_a)
    b_elo = state.elo_for(player_b)
    elo_diff = a_elo - b_elo
    elo_prob = _elo_probability(a_elo, b_elo)

    # --- H2H ---
    h2h_obj = state.h2h(player_a, player_b)
    h2h_f = h2h_obj.as_features(player_a, state.defaults)
    h2h_matches = int(h2h_f["h2h_matches"])

    # --- upset-risk heuristics (simplified, using ML's own features) ---
    upset_risk_flags = []
    if max_winner_prob < 0.58:
        upset_risk_flags.append("weak winner edge")
    if h2h_matches >= 6 and ((player_a_win_probability >= 0.5 and h2h_f["h2h_a_win_rate"] < 0.40)
                              or (player_a_win_probability < 0.5 and h2h_f["h2h_a_win_rate"] > 0.60)):
        upset_risk_flags.append("H2H conflict")
    upset_risk = "Low" if not upset_risk_flags and max_winner_prob >= 0.62 else \
                 "Medium" if len(upset_risk_flags) <= 1 else "High"

    return {
        "player_a": player_a,
        "player_b": player_b,
        "player_a_win_probability": player_a_win_probability,
        "player_b_win_probability": player_b_win_probability,
        "predicted_winner": player_a if player_a_win_probability >= 0.5 else player_b,
        "elo_probability": elo_prob,
        "elo_a": a_elo,
        "elo_b": b_elo,
        "elo_diff": elo_diff,
        "h2h_matches": h2h_matches,
        "h2h_player_a_wins": int(h2h_f["h2h_a_win_rate"] * h2h_f["h2h_matches"]),
        "h2h_player_b_wins": h2h_matches - int(h2h_f["h2h_a_win_rate"] * h2h_f["h2h_matches"]),
        "h2h_player_a_win_rate": h2h_f["h2h_a_win_rate"],
        "upset_risk": upset_risk,
        "upset_risk_flags": upset_risk_flags,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "first_set_over_probability": first_over_prob,
        "first_set_under_probability": first_under_prob,
        "first_set_line": float(first_set_line),
        "expected_first_set_points": first_pts_pred,
        "total_points_over_probability": total_over_prob,
        "total_points_under_probability": total_under_prob,
        "total_points_line": float(total_points_line),
        "expected_total_points": total_pred,
        "expected_sets_played": 0.0,
        "sets_line": float(sets_line),
        "sets_over_probability": 0.5,
        "sets_under_probability": 0.5,
        "features": features,
    }


def save_model_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path):
    return joblib.load(Path(path))
