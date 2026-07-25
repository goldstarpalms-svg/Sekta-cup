from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.backtesting import run_holdout_backtest, threshold_table
from src.github_storage import github_storage_enabled
from src.persistence import (
    DAILY_RESULTS_FILE,
    STRONG_PICKS_FILE,
    load_daily_results,
    load_strong_picks,
    official_results_to_match_history,
    reset_daily_results,
    reset_strong_picks,
    save_daily_results,
    save_strong_picks,
)
from src.setka_core import (
    build_context,
    comparison_table,
    format_number,
    format_percent,
    get_head_to_head,
    load_raw_data,
    predict_match,
)

try:
    from src.ml_pipeline import (
        load_model_bundle,
        metrics_table,
        predict_with_bundle,
        save_model_bundle,
        train_model_bundle,
    )

    ML_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - shown inside the UI
    ML_IMPORT_ERROR = exc

try:
    from src.odds_api import (
        OddsAPIError,
        add_implied_probabilities,
        fetch_odds,
        list_sports,
        normalize_odds_events,
    )
    from src.setka_live import (
        OFFICIAL_SETKA_URL,
        add_lagos_time,
        add_location_names,
        fetch_live_matches,
        fetch_nearest_matches,
        fetch_official_site_status,
        fetch_results_for_date,
        location_map,
        status_as_dict,
    )
    from src.source_registry import categories as source_categories
    from src.source_registry import registry_dataframe, summary_by_category

    ODDS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - shown inside the UI
    ODDS_IMPORT_ERROR = exc


st.set_page_config(
    page_title="Setka Predictor",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional permanent storage via GitHub. Add these in Streamlit secrets:
# GITHUB_STORAGE_TOKEN, GITHUB_STORAGE_REPO, GITHUB_STORAGE_BRANCH, GITHUB_STORAGE_PREFIX
try:
    for key in [
        "GITHUB_STORAGE_TOKEN",
        "GITHUB_STORAGE_REPO",
        "GITHUB_STORAGE_BRANCH",
        "GITHUB_STORAGE_PREFIX",
    ]:
        value = st.secrets.get(key)
        if value:
            os.environ[key] = str(value)
except Exception:
    pass


CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.small-muted { color: #94A3B8; font-size: 0.92rem; }
.card {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    background: rgba(15, 23, 42, 0.48);
}
.good { color: #22C55E; font-weight: 700; }
.warn { color: #F59E0B; font-weight: 700; }
.bad { color: #EF4444; font-weight: 700; }
.pick-card {
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.42));
}
.pick-title { font-size: 1rem; font-weight: 800; margin-bottom: 0.25rem; }
.pick-meta { color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.55rem; }
.hero {
    border-radius: 26px;
    padding: 2.0rem 1.4rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(248, 250, 252, 0.10);
    background: radial-gradient(circle at top left, rgba(249, 115, 22, 0.35), transparent 30%), linear-gradient(135deg, #07111f, #111827 48%, #1e293b);
}
.hero h1 { font-size: clamp(2rem, 6vw, 4.2rem); line-height: 1.0; margin-bottom: 0.45rem; }
.hero p { color: #CBD5E1; font-size: 1.05rem; max-width: 760px; }
.feature-card {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    padding: 1rem;
    min-height: 145px;
    background: rgba(15, 23, 42, 0.55);
}
.feature-card h3 { margin-top: 0; margin-bottom: 0.35rem; }
.badge-strong { color: #22C55E; font-weight: 900; }
.badge-medium { color: #F59E0B; font-weight: 900; }
.badge-watch { color: #38BDF8; font-weight: 900; }
.badge-avoid { color: #EF4444; font-weight: 900; }
.terminal-panel {
    border: 1px solid rgba(34, 197, 94, 0.28);
    border-radius: 22px;
    padding: 1rem;
    background: linear-gradient(135deg, rgba(2, 6, 23, 0.92), rgba(15, 23, 42, 0.75));
    box-shadow: 0 0 25px rgba(34, 197, 94, 0.08);
    margin-bottom: 0.8rem;
}
.green-card { border-left: 5px solid #22C55E; }
.watch-card { border-left: 5px solid #F59E0B; }
.nobet-card { border-left: 5px solid #EF4444; }
.terminal-title { font-size: 1.1rem; font-weight: 900; }
.terminal-value { font-size: 1.35rem; font-weight: 900; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and preparing Setka data...")
def load_app_context() -> dict:
    matches, leaderboard = load_raw_data()
    saved_results = load_daily_results()
    extra_matches = official_results_to_match_history(saved_results)
    if not extra_matches.empty:
        matches = pd.concat([matches, extra_matches], ignore_index=True)
        matches = matches.drop_duplicates(subset=["source_match_id"], keep="last")
    context = build_context(matches, leaderboard)
    context["extra_saved_result_matches"] = len(extra_matches)
    return context


ctx = load_app_context()
matches = ctx["matches"]
player_log = ctx["player_log"]
player_stats = ctx["player_stats"]
global_stats = ctx["global_stats"]

players_by_elo = player_stats.sort_values(["elo", "matches"], ascending=[False, False])[
    "player"
].tolist()
players_alpha = sorted(player_stats["player"].dropna().unique().tolist())
MODEL_BUNDLE_PATH = Path("models/setka_ml_bundle.joblib")


@st.cache_resource(show_spinner="Training ML models. This can take a few minutes on the full dataset...")
def train_ml_cached(algorithm: str, max_training_rows: int | None):
    return train_model_bundle(
        matches,
        algorithm=algorithm,
        max_training_rows=max_training_rows,
    )


@st.cache_data(show_spinner="Running time-split backtest...")
def run_backtest_cached(test_rows: int, first_set_line: float, total_points_line: float, sets_line: float):
    raw_matches, raw_leaderboard = load_raw_data()
    return run_holdout_backtest(
        raw_matches,
        raw_leaderboard,
        test_rows=test_rows,
        first_set_line=first_set_line,
        total_points_line=total_points_line,
        sets_line=sets_line,
    )


with st.sidebar:
    st.title("🏓 Setka Predictor")
    st.caption("Prediction dashboard from uploaded Setka match history + Elo leaderboard.")
    st.divider()

    page = st.radio(
        "Go to",
        [
            "Home",
            "Setka Trading Desk",
            "Live Predictions",
            "Live Match Center",
            "Owner Edge Engine",
            "Strong Pick Tracker",
            "Results Checker",
            "Match Predictor",
            "Accuracy Lab",
            "Smart Stake Calc",
            "Bet Slip Tools",
            "ML Lab",
            "Live Odds",
            "Data Sources",
            "Leaderboard",
            "Player Explorer",
            "Head-to-Head",
            "Data Health",
        ],
    )

    st.divider()
    st.metric("Matches", f"{global_stats['match_count']:,}")
    st.metric("Players", f"{global_stats['player_count']:,}")
    st.caption(
        f"Date range: {global_stats['date_min'].date()} → {global_stats['date_max'].date()}"
    )
    st.caption(
        "Rule model: Elo + form + H2H. ML Lab: scikit-learn/XGBoost training."
    )
    if ctx.get("extra_saved_result_matches", 0):
        st.caption(f"Saved official results added to model context: {ctx['extra_saved_result_matches']:,}")


def probability_bar(pred: dict) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                y=[pred["player_a"], pred["player_b"]],
                x=[pred["player_a_win_probability"], pred["player_b_win_probability"]],
                orientation="h",
                marker_color=["#22C55E", "#F97316"],
                text=[
                    format_percent(pred["player_a_win_probability"]),
                    format_percent(pred["player_b_win_probability"]),
                ],
                textposition="auto",
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Win probability",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def over_under_bar(label: str, over_prob: float, under_prob: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Over", "Under"],
                y=[over_prob, under_prob],
                marker_color=["#38BDF8", "#A78BFA"],
                text=[format_percent(over_prob), format_percent(under_prob)],
                textposition="auto",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=label,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def h2h_display_table(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    cols = [
        "date_time",
        "competition",
        "player1",
        "player2",
        "winner",
        "set_scores",
        "total_points",
        "first_set_total",
        "sets_played",
    ]
    out = df.loc[:, [c for c in cols if c in df.columns]].head(limit).copy()
    if "date_time" in out:
        out["date_time"] = pd.to_datetime(out["date_time"]).dt.strftime("%Y-%m-%d %H:%M")
    return out


def player_latest_table(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    cols = [
        "date_time",
        "competition",
        "opponent",
        "won",
        "set_scores",
        "points_for",
        "points_against",
        "total_points",
        "first_set_total",
    ]
    out = df.loc[:, [c for c in cols if c in df.columns]].sort_values(
        "date_time", ascending=False
    ).head(limit).copy()
    out["result"] = out["won"].map({True: "Win", False: "Loss"})
    out = out.drop(columns=["won"])
    out["date_time"] = pd.to_datetime(out["date_time"]).dt.strftime("%Y-%m-%d %H:%M")
    return out


@st.cache_data(ttl=30, show_spinner="Fetching official Setka upcoming matches...")
def load_official_nearest() -> pd.DataFrame:
    locs = location_map()
    frame = fetch_nearest_matches()
    frame = add_lagos_time(frame)
    frame = add_location_names(frame, locs)
    return frame


@st.cache_data(ttl=15, show_spinner="Fetching official Setka live matches...")
def load_official_live() -> pd.DataFrame:
    locs = location_map()
    frame = fetch_live_matches()
    frame = add_lagos_time(frame)
    frame = add_location_names(frame, locs)
    return frame


@st.cache_data(ttl=45, show_spinner="Fetching official Setka results...")
def load_official_results(match_date: str, day_period: int | None) -> pd.DataFrame:
    locs = location_map()
    frame = fetch_results_for_date(match_date, day_period=day_period)
    frame = add_lagos_time(frame)
    frame = add_location_names(frame, locs)
    return frame


def prediction_pick_row(row: pd.Series, first_set_line: float, total_points_line: float, sets_line: float) -> dict:
    pred = predict_match(
        row["player1"],
        row["player2"],
        player_stats,
        matches,
        global_stats,
        first_set_line=first_set_line,
        total_points_line=total_points_line,
        sets_line=sets_line,
    )
    winner_prob = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
    total_pick = "Over" if pred["total_points_over_probability"] >= pred["total_points_under_probability"] else "Under"
    total_prob = max(pred["total_points_over_probability"], pred["total_points_under_probability"])
    first_pick = "Over" if pred["first_set_over_probability"] >= pred["first_set_under_probability"] else "Under"
    first_prob = max(pred["first_set_over_probability"], pred["first_set_under_probability"])
    sets_pick = "Over" if pred["sets_over_probability"] >= pred["sets_under_probability"] else "Under"
    sets_prob = max(pred["sets_over_probability"], pred["sets_under_probability"])
    return {
        "match_id": row.get("match_id"),
        "time_lagos": row.get("start_time_lagos"),
        "date_lagos": row.get("start_date_lagos"),
        "location": row.get("location"),
        "match": f"{row['player1']} vs {row['player2']}",
        "player1": row["player1"],
        "player2": row["player2"],
        "winner_pick": pred["predicted_winner"],
        "winner_probability": winner_prob,
        "total_pick": total_pick,
        "total_probability": total_prob,
        "expected_total_points": pred["expected_total_points"],
        "total_points_line": float(total_points_line),
        "first_set_pick": first_pick,
        "first_set_probability": first_prob,
        "expected_first_set_points": pred["expected_first_set_points"],
        "first_set_line": float(first_set_line),
        "sets_pick": sets_pick,
        "sets_probability": sets_prob,
        "expected_sets_played": pred["expected_sets_played"],
        "sets_line": pred["sets_line"],
        "confidence": pred["confidence"],
        "confidence_score": pred["confidence_score"],
        "upset_risk": pred.get("upset_risk", ""),
        "upset_risk_flags": ", ".join(pred.get("upset_risk_flags", [])),
        "h2h_matches": pred["h2h_matches"],
    }


def pick_strength(probability: float | None, confidence: str | None = None, upset_risk: str | None = None) -> str:
    if probability is None or pd.isna(probability):
        return "Avoid"
    p = float(probability)
    confidence = confidence or ""
    upset_risk = upset_risk or ""
    if upset_risk == "High" and p < 0.70:
        return "Avoid"
    if upset_risk == "Medium" and p < 0.62:
        return "Watch"
    if p >= 0.68 and confidence == "High" and upset_risk != "High":
        return "Strong"
    if p >= 0.60:
        return "Medium"
    if p >= 0.55:
        return "Watch"
    return "Avoid"


def strength_class(strength: str) -> str:
    return {
        "Strong": "badge-strong",
        "Medium": "badge-medium",
        "Watch": "badge-watch",
        "Avoid": "badge-avoid",
    }.get(strength, "badge-avoid")


def apply_pick_strengths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["winner_strength"] = out.apply(
        lambda r: pick_strength(r.get("winner_probability"), r.get("confidence"), r.get("upset_risk")), axis=1
    )
    out["total_strength"] = out.apply(
        lambda r: pick_strength(r.get("total_probability"), r.get("confidence"), r.get("upset_risk")), axis=1
    )
    out["first_set_strength"] = out.apply(
        lambda r: pick_strength(r.get("first_set_probability"), r.get("confidence"), r.get("upset_risk")), axis=1
    )
    out["sets_strength"] = out.apply(
        lambda r: pick_strength(r.get("sets_probability"), r.get("confidence"), r.get("upset_risk")), axis=1
    )
    out["best_market"] = out[["winner_probability", "total_probability", "first_set_probability", "sets_probability"]].idxmax(axis=1)
    out["best_market"] = out["best_market"].map(
        {
            "winner_probability": "Winner",
            "total_probability": "Total",
            "first_set_probability": "1st Set",
            "sets_probability": "Sets",
        }
    )
    out["best_probability"] = out[["winner_probability", "total_probability", "first_set_probability", "sets_probability"]].max(axis=1)
    out["best_pick"] = out.apply(
        lambda r: r["winner_pick"]
        if r["best_market"] == "Winner"
        else f"{r['total_pick']} total"
        if r["best_market"] == "Total"
        else f"{r['first_set_pick']} 1st set"
        if r["best_market"] == "1st Set"
        else f"{r['sets_pick']} sets",
        axis=1,
    )
    out["best_strength"] = out.apply(
        lambda r: pick_strength(r.get("best_probability"), r.get("confidence"), r.get("upset_risk")), axis=1
    )
    out["snapshot_time_lagos"] = pd.Timestamp.now(tz="Africa/Lagos").strftime("%Y-%m-%d %H:%M:%S")
    return out


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["winner_probability", "total_probability", "first_set_probability", "sets_probability", "best_probability"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for col in ["expected_total_points", "expected_first_set_points", "expected_sets_played", "confidence_score"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    return out


def render_mobile_pick_cards(df: pd.DataFrame, limit: int = 8) -> None:
    for _, r in df.head(limit).iterrows():
        strength = r.get("best_strength", "Avoid")
        css_class = strength_class(strength)
        st.markdown(
            f"""
<div class="pick-card">
  <div class="pick-title">{r.get('time_lagos', '')} • {r.get('match', '')}</div>
  <div class="pick-meta">{r.get('location', '')} • H2H {r.get('h2h_matches', 0)} • Confidence {r.get('confidence', '')} • Upset risk {r.get('upset_risk', '')}</div>
  <div>Best: <b>{r.get('best_market', '')}</b> — <b>{r.get('best_pick', '')}</b> ({r.get('best_probability', 0):.1%}) <span class="{css_class}">{strength}</span></div>
  <div class="pick-meta">Winner: {r.get('winner_pick', '')} {r.get('winner_probability', 0):.1%} • Total: {r.get('total_pick', '')} {r.get('total_probability', 0):.1%} • 1st set: {r.get('first_set_pick', '')} {r.get('first_set_probability', 0):.1%} • Sets: {r.get('sets_pick', '')} {r.get('sets_probability', 0):.1%}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def enable_browser_auto_refresh(seconds: int) -> None:
    if seconds <= 0:
        return
    components.html(
        f"""
<script>
  setTimeout(function() {{ window.parent.location.reload(); }}, {int(seconds) * 1000});
</script>
""",
        height=0,
    )


def decimal_from_text(value: str) -> float | None:
    try:
        number = float(str(value).strip())
        return number if number > 1 else None
    except Exception:
        return None


def parse_decimal_odds(text: str) -> list[float]:
    return [x for x in (decimal_from_text(v) for v in re.split(r"[\s,;|/]+", text or "")) if x]


def combined_decimal_odds(odds: list[float]) -> float:
    value = 1.0
    for odd in odds:
        value *= float(odd)
    return value


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - probability
    return max(0.0, ((b * probability) - q) / b)


def implied_probability(decimal_odds: float) -> float:
    return 1 / decimal_odds if decimal_odds and decimal_odds > 1 else 0.0


def fair_decimal(probability: float) -> float | None:
    if probability is None or pd.isna(probability) or probability <= 0:
        return None
    return 1 / float(probability)


def minimum_value_odds(probability: float, edge_buffer: float = 0.03) -> float | None:
    if probability is None or pd.isna(probability):
        return None
    usable_probability = max(0.01, float(probability) - float(edge_buffer))
    return 1 / usable_probability


def owner_edge_score(row: pd.Series) -> float:
    prob = float(row.get("best_probability", 0) or 0)
    confidence_bonus = {"High": 0.08, "Medium": 0.03, "Low": -0.05}.get(str(row.get("confidence", "")), 0)
    risk_penalty = {"Low": 0.00, "Medium": 0.06, "High": 0.18}.get(str(row.get("upset_risk", "")), 0.08)
    h2h_bonus = min(float(row.get("h2h_matches", 0) or 0), 60) / 1000
    strength_bonus = {"Strong": 0.04, "Medium": 0.01, "Watch": -0.02, "Avoid": -0.10}.get(str(row.get("best_strength", "")), 0)
    return max(0.0, min(1.0, prob + confidence_bonus + h2h_bonus + strength_bonus - risk_penalty))


def owner_decision(row: pd.Series) -> str:
    score = float(row.get("edge_score", 0) or 0)
    prob = float(row.get("best_probability", 0) or 0)
    if row.get("best_market") == "Winner" and row.get("upset_risk") == "High":
        return "NO BET"
    if score >= 0.72 and prob >= 0.60:
        return "GREEN"
    if score >= 0.64 and prob >= 0.57:
        return "WATCH"
    return "NO BET"


def owner_reason(row: pd.Series) -> str:
    reasons = []
    if row.get("confidence") == "High":
        reasons.append("high confidence")
    if float(row.get("h2h_matches", 0) or 0) >= 10:
        reasons.append("useful H2H sample")
    if row.get("upset_risk") == "Low":
        reasons.append("low upset risk")
    if row.get("best_strength") in ["Strong", "Medium"]:
        reasons.append(f"{row.get('best_strength')} model strength")
    if not reasons:
        reasons.append("price-dependent only")
    return ", ".join(reasons)


FORMER_PREDICTIONS = {
    804491: {"winner_pick": "Dmitri Gribcov", "total_pick": "Over", "first_set_pick": "Over"},
    804612: {"winner_pick": "Orest Hura", "total_pick": "Over", "first_set_pick": "Over"},
    804655: {"winner_pick": "Yan Krol", "total_pick": "Over", "first_set_pick": "Over"},
    804627: {"winner_pick": "Oleh Lutsyshyn", "total_pick": "Under", "first_set_pick": "Over"},
    804640: {"winner_pick": "Yevhen Kryvorotko", "total_pick": "Under", "first_set_pick": "Under"},
    804492: {"winner_pick": "Mihail Filip", "total_pick": "Under", "first_set_pick": "Over"},
    804613: {"winner_pick": "Anton Shypilov", "total_pick": "Over", "first_set_pick": "Over"},
    804656: {"winner_pick": "Vitalii Khamurda", "total_pick": "Over", "first_set_pick": "Over"},
    804628: {"winner_pick": "Serhii Prus", "total_pick": "Under", "first_set_pick": "Over"},
    804641: {"winner_pick": "Serhei Hohenko", "total_pick": "Under", "first_set_pick": "Over"},
}


def grade_pick(prediction: str | None, actual: str | None) -> str:
    if not prediction or not actual:
        return "Pending"
    return "✅" if str(prediction).strip() == str(actual).strip() else "❌"


def save_strong_picks_to_tracker(frame: pd.DataFrame, source: str) -> int:
    """Persist GREEN/strong picks until the user resets the tracker."""
    if frame is None or frame.empty:
        return 0
    saved = frame.copy()
    saved["saved_at_lagos"] = pd.Timestamp.now(tz="Africa/Lagos").strftime("%Y-%m-%d %H:%M:%S")
    saved["tracker_source"] = source
    before = len(load_strong_picks())
    combined = save_strong_picks(saved)
    st.session_state["strong_pick_tracker"] = combined
    return max(0, len(combined) - before)


def expand_setka_result_dates(date_texts: list[str]) -> list[str]:
    """Setka night tournaments can map to previous/next official dates.

    If a saved pick is Lagos date 2026-07-25, official tournament/result
    rows may be under 2026-07-24 or 2026-07-25 depending on location/period.
    So the tracker checks date -1, date, and date +1.
    """
    dates: set[str] = set()
    for value in date_texts:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            continue
        base = dt.date()
        for offset in [-1, 0, 1]:
            dates.add((pd.Timestamp(base) + pd.Timedelta(days=offset)).date().isoformat())
    return sorted(dates)


def tracker_result_dates(track: pd.DataFrame, manual_date) -> list[str]:
    """Build a robust official Setka date search window for saved picks."""
    seeds: list[str] = [str(manual_date), pd.Timestamp.now(tz="Africa/Lagos").date().isoformat()]
    for col in ["date_lagos", "saved_at_lagos"]:
        if col in track.columns:
            parsed = pd.to_datetime(track[col], errors="coerce").dropna()
            seeds.extend([x.date().isoformat() for x in parsed.tolist()])
    dates: set[str] = set()
    for seed in seeds:
        dt = pd.to_datetime(seed, errors="coerce")
        if pd.isna(dt):
            continue
        base = dt.date()
        # Wider than before because Setka night events can sit on adjacent official days.
        for offset in range(-3, 4):
            dates.add((pd.Timestamp(base) + pd.Timedelta(days=offset)).date().isoformat())
    return sorted(dates)


def fetch_tracker_results(track: pd.DataFrame, manual_date) -> pd.DataFrame:
    dates = tracker_result_dates(track, manual_date)
    result_df = sync_official_results_for_dates(dates)
    # Add the current live widget too; it often contains just-finished/live rows before tournament endpoint catches up.
    try:
        live_now = load_official_live()
        if live_now is not None and not live_now.empty:
            live_now = live_now.copy()
            live_now["synced_at_lagos"] = pd.Timestamp.now(tz="Africa/Lagos").strftime("%Y-%m-%d %H:%M:%S")
            result_df = pd.concat([result_df, live_now], ignore_index=True) if not result_df.empty else live_now
            save_daily_results(live_now)
    except Exception:
        pass
    if result_df is None or result_df.empty:
        return pd.DataFrame()
    if "match_id" in result_df.columns:
        result_df["match_id"] = pd.to_numeric(result_df["match_id"], errors="coerce")
        result_df = result_df.dropna(subset=["match_id"])
        result_df["match_id"] = result_df["match_id"].astype("int64")
        result_df = result_df.drop_duplicates(subset=["match_id"], keep="last")
    return result_df


def sync_official_results_for_dates(date_texts: list[str]) -> pd.DataFrame:
    """Fetch official Setka results for dates and persist them for future grading/training export."""
    frames = []
    for date_text in date_texts:
        try:
            frame = load_official_results(str(date_text), None)
            if frame is not None and not frame.empty:
                frame = frame.copy()
                frame["synced_at_lagos"] = pd.Timestamp.now(tz="Africa/Lagos").strftime("%Y-%m-%d %H:%M:%S")
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        return load_daily_results()
    combined_new = pd.concat(frames, ignore_index=True)
    return save_daily_results(combined_new)


def grade_best_pick(row: pd.Series) -> tuple[str, str | None]:
    market = str(row.get("best_market", ""))
    pick = str(row.get("best_pick", ""))
    if pd.isna(row.get("status")) or str(row.get("status")) not in ["Finished", "Technical"]:
        return "Pending", None
    if market == "Winner":
        actual = row.get("winner") or None
        return grade_pick(row.get("winner_pick"), actual), actual
    if market == "Total":
        line = float(row.get("total_points_line", 75.5) or 75.5)
        if pd.isna(row.get("total_points")):
            return "Pending", None
        actual = "Over" if float(row.get("total_points")) > line else "Under"
        return grade_pick(row.get("total_pick"), actual), f"{actual} {line}"
    if market == "1st Set":
        line = float(row.get("first_set_line", 18.5) or 18.5)
        if pd.isna(row.get("first_set_total")):
            return "Pending", None
        actual = "Over" if float(row.get("first_set_total")) > line else "Under"
        return grade_pick(row.get("first_set_pick"), actual), f"{actual} {line}"
    if market == "Sets":
        line = float(row.get("sets_line", 3.5) or 3.5)
        if pd.isna(row.get("sets_played")):
            return "Pending", None
        actual = "Over" if float(row.get("sets_played")) > line else "Under"
        return grade_pick(row.get("sets_pick"), actual), f"{actual} {line}"
    return "Pending", None


if page == "Home":
    st.markdown(
        """
<div class="hero">
  <div class="small-muted">🏓 SETKA CUP TABLE-TENNIS PREDICTION DASHBOARD</div>
  <h1>Predict Setka smarter with data.</h1>
  <p>Focused on Setka Cup table tennis: live fixtures, winner picks, total-points, first-set Over/Under 18.5, result checking, player stats, and H2H analysis.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical Setka matches", f"{global_stats['match_count']:,}")
    m2.metric("Players tracked", f"{global_stats['player_count']:,}")
    m3.metric("Main sport", "Table Tennis")
    m4.metric("Main timezone", "Lagos")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><h3>💎 Owner Edge Engine</h3><p>A strict no-bet-first control room that gives minimum value odds, bankroll caps, and only GREEN/WATCH/NO BET decisions.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><h3>✅ Setka Result Checker</h3><p>Official Setka scores, set totals, live status, and grading for prediction CSV snapshots.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><h3>🏓 Table Tennis Predictor</h3><p>Player stats, leaderboard, head-to-head, expected total points, and first-set Over/Under 18.5.</p></div>', unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown('<div class="feature-card"><h3>🧮 Smart Stake Calc</h3><p>Optional table-tennis stake planner with payout, implied probability, model edge, EV, and safer stake sizing.</p></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="feature-card"><h3>🎟️ Bet Slip Tools</h3><p>Optional odds slip calculator for table-tennis selections: combined odds, payout, and risk level.</p></div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="feature-card"><h3>🎯 Accuracy Lab</h3><p>Backtest recent Setka matches, discover stronger probability filters, and reduce weak picks.</p></div>', unsafe_allow_html=True)

    st.info("This app is for analytical decision support. It does not guarantee results and is not financial advice.")


elif page == "Setka Trading Desk":
    st.title("💎 Setka Trading Desk")
    st.markdown("Premium owner terminal: live ticker, best value picks, NO BET zone, bankroll protection, and model health.")

    st.error("If you are losing back-to-back: stop chasing. Use this page in Protection Mode and accept NO BET when the edge is not clean.")

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Live dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.caption("Trading Desk controls")
        desk_mode = st.selectbox("Mode", ["Protection Mode", "Balanced Mode", "Aggressive Mode"], index=0)
        desk_auto_refresh = st.checkbox("Auto-refresh desk", value=False)
        desk_refresh_seconds = st.selectbox("Desk refresh every", [10, 15, 30, 60], index=1)
    if desk_auto_refresh:
        enable_browser_auto_refresh(desk_refresh_seconds)

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        desk_bankroll = st.number_input("Bankroll", min_value=0.0, value=10000.0, step=500.0, key="desk_bankroll")
    with d2:
        daily_loss_limit_pct = st.slider("Daily stop-loss %", 1, 30, 10, 1, key="desk_loss_limit") / 100
    with d3:
        losses_today = st.number_input("Losses today", min_value=0.0, value=0.0, step=100.0, key="desk_losses")
    with d4:
        daily_risk_cap_pct = st.slider("Daily risk cap %", 1, 30, 8, 1, key="desk_risk_cap") / 100
    with d5:
        if st.button("Refresh desk", type="primary"):
            st.cache_data.clear()
            st.rerun()

    if desk_mode == "Protection Mode":
        min_green_score, min_green_prob, required_edge_buffer, max_pick_stake_pct = 0.76, 0.62, 0.05, 0.005
    elif desk_mode == "Balanced Mode":
        min_green_score, min_green_prob, required_edge_buffer, max_pick_stake_pct = 0.72, 0.60, 0.03, 0.010
    else:
        min_green_score, min_green_prob, required_edge_buffer, max_pick_stake_pct = 0.68, 0.58, 0.02, 0.015

    stop_loss_amount = desk_bankroll * daily_loss_limit_pct
    daily_risk_cap = desk_bankroll * daily_risk_cap_pct
    stop_active = losses_today >= stop_loss_amount and stop_loss_amount > 0
    remaining_loss_room = max(0.0, stop_loss_amount - losses_today)

    try:
        live_feed = load_official_live()
    except Exception:
        live_feed = pd.DataFrame()
    try:
        upcoming = load_official_nearest().head(30)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    # Live ticker
    st.subheader("🔴 Live ticker")
    if live_feed.empty:
        st.info("No live matches returned right now.")
    else:
        live_view = live_feed.copy()
        live_view["match"] = live_view["player1"] + " vs " + live_view["player2"]
        live_view["score"] = live_view["player1_score"].fillna(0).astype(int).astype(str) + " - " + live_view["player2_score"].fillna(0).astype(int).astype(str)
        ticker_cols = ["start_time_lagos", "location", "status", "match", "score", "set_scores", "active_player"]
        st.dataframe(live_view[[c for c in ticker_cols if c in live_view.columns]].head(12), use_container_width=True, height=260)

    # Upcoming edge scan
    rows = []
    for _, match_row in upcoming.iterrows():
        if match_row.get("player1") and match_row.get("player2"):
            rows.append(prediction_pick_row(match_row, 18.5, 75.5, 3.5))
    desk_df = apply_pick_strengths(pd.DataFrame(rows))
    if desk_df.empty:
        st.warning("No upcoming matches available for desk scan.")
        st.stop()

    desk_df["edge_score"] = desk_df.apply(owner_edge_score, axis=1)
    desk_df["fair_odds"] = desk_df["best_probability"].map(fair_decimal)
    desk_df["minimum_value_odds"] = desk_df["best_probability"].map(lambda p: minimum_value_odds(p, required_edge_buffer))
    desk_df["reason"] = desk_df.apply(owner_reason, axis=1)

    def desk_decision(row):
        if stop_active:
            return "STOP"
        if row.get("best_market") == "Winner" and row.get("upset_risk") == "High":
            return "NO BET"
        if row.get("edge_score", 0) >= min_green_score and row.get("best_probability", 0) >= min_green_prob and row.get("best_strength") in ["Strong", "Medium"]:
            return "GREEN"
        if row.get("edge_score", 0) >= (min_green_score - 0.08) and row.get("best_probability", 0) >= 0.56:
            return "WATCH"
        return "NO BET"

    desk_df["desk_decision"] = desk_df.apply(desk_decision, axis=1)
    desk_df["stake_cap"] = desk_df.apply(
        lambda r: 0.0 if r["desk_decision"] != "GREEN" else min(desk_bankroll * max_pick_stake_pct, daily_risk_cap, remaining_loss_room if remaining_loss_room > 0 else desk_bankroll * max_pick_stake_pct),
        axis=1,
    )

    green = desk_df.loc[desk_df["desk_decision"] == "GREEN"].sort_values("edge_score", ascending=False)
    watch = desk_df.loc[desk_df["desk_decision"] == "WATCH"].sort_values("edge_score", ascending=False)
    nobet = desk_df.loc[desk_df["desk_decision"].isin(["NO BET", "STOP"])]

    st.subheader("🛡️ Bankroll protection")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Mode", desk_mode)
    p2.metric("Stop-loss", f"{stop_loss_amount:,.2f}")
    p3.metric("Loss room left", f"{remaining_loss_room:,.2f}")
    p4.metric("Daily risk cap", f"{daily_risk_cap:,.2f}")
    p5.metric("Status", "STOP" if stop_active else "SAFE")
    if stop_active:
        st.error("STOP MODE ACTIVE: Daily loss limit reached. The desk will not recommend new stakes.")

    st.subheader("💚 GREEN opportunities")
    cols = ["time_lagos", "location", "match", "best_market", "best_pick", "best_probability", "edge_score", "minimum_value_odds", "stake_cap", "confidence", "upset_risk", "reason"]
    green_display = green[cols].copy()
    for c in ["best_probability", "edge_score"]:
        green_display[c] = green_display[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for c in ["minimum_value_odds", "stake_cap"]:
        green_display[c] = green_display[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if green_display.empty:
        st.info("No GREEN opportunities right now. In Protection Mode this is normal. Wait for cleaner spots.")
    else:
        st.dataframe(green_display, use_container_width=True, height=300)
        if st.button("Save GREEN picks to Strong Pick Tracker", key="save_trading_green"):
            count = save_strong_picks_to_tracker(green, "Setka Trading Desk")
            st.success(f"Saved {count} GREEN picks to Strong Pick Tracker.")

    st.subheader("🧠 Model health")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Scanned", f"{len(desk_df):,}")
    h2.metric("GREEN", f"{len(green):,}")
    h3.metric("WATCH", f"{len(watch):,}")
    h4.metric("NO BET/STOP", f"{len(nobet):,}")

    with st.expander("WATCH and NO BET board", expanded=False):
        full_cols = ["time_lagos", "location", "match", "best_market", "best_pick", "best_probability", "edge_score", "desk_decision", "minimum_value_odds", "confidence", "upset_risk", "upset_risk_flags", "reason"]
        full = desk_df[full_cols].sort_values(["desk_decision", "edge_score"], ascending=[True, False]).copy()
        for c in ["best_probability", "edge_score"]:
            full[c] = full[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        full["minimum_value_odds"] = full["minimum_value_odds"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(full, use_container_width=True, height=520)

    st.download_button("Download Trading Desk CSV", desk_df.to_csv(index=False).encode("utf-8"), "setka_trading_desk.csv", "text/csv")
    st.caption("Professional rule: no odds value, no bet. Stop-loss reached, no bet. Weak edge, no bet.")


elif page == "Live Predictions":
    st.title("🔴 Live Setka Predictions")
    st.markdown("Fetch upcoming matches from the official Setka API and generate winner, total-points, and first-set picks in Lagos time.")

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Live dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.caption("Live prediction filters")
        live_limit = st.slider("Upcoming matches to read", 5, 50, 20, 5)
        min_winner = st.slider("Min winner probability", 50, 90, 60, 1) / 100
        min_total = st.slider("Min total-points probability", 50, 80, 57, 1) / 100
        min_first = st.slider("Min first-set probability", 50, 80, 56, 1) / 100
        min_sets = st.slider("Min sets O/U probability", 50, 85, 58, 1) / 100
        hide_high_upset = st.checkbox("Hide high upset-risk winner picks", value=True)
        show_only_strong = st.checkbox("Show only picks passing filters", value=False)
        auto_refresh = st.checkbox("Auto-refresh page", value=False)
        refresh_seconds = st.selectbox("Refresh every", [15, 30, 60, 120], index=1)

    if auto_refresh:
        enable_browser_auto_refresh(refresh_seconds)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        first_set_line = st.number_input("1st set line", 10.5, 35.5, 18.5, 0.5, key="live_first")
    with c2:
        total_points_line = st.number_input("Total points line", 30.5, 140.5, 75.5, 0.5, key="live_total")
    with c3:
        sets_line = st.selectbox("Sets line", [3.5, 4.5], index=0, key="live_sets")
    with c4:
        if st.button("Refresh official feed", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c5:
        st.caption("Cache refreshes automatically every 30 seconds.")

    try:
        upcoming = load_official_nearest().head(live_limit)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    rows = []
    for _, match_row in upcoming.iterrows():
        if match_row.get("player1") and match_row.get("player2"):
            rows.append(prediction_pick_row(match_row, first_set_line, total_points_line, sets_line))
    pred_df = apply_pick_strengths(pd.DataFrame(rows))

    if pred_df.empty:
        st.warning("No upcoming official Setka matches returned right now.")
        st.stop()

    filtered = pred_df.copy()
    if show_only_strong:
        filtered = filtered.loc[
            (filtered["winner_probability"] >= min_winner)
            | (filtered["total_probability"] >= min_total)
            | (filtered["first_set_probability"] >= min_first)
            | (filtered["sets_probability"] >= min_sets)
        ]
    if hide_high_upset:
        filtered = filtered.loc[~((filtered["best_market"] == "Winner") & (filtered["upset_risk"] == "High"))]

    top_cols = st.columns(5)
    top_cols[0].metric("Official matches", f"{len(upcoming):,}")
    top_cols[1].metric("Shown after filter", f"{len(filtered):,}")
    top_cols[2].metric("Strong best picks", f"{(pred_df['best_strength'] == 'Strong').sum():,}")
    top_cols[3].metric("High confidence", f"{(pred_df['confidence'] == 'High').sum():,}")
    top_cols[4].metric("Avg H2H", f"{pred_df['h2h_matches'].mean():.1f}")

    st.subheader("Best picks cards")
    best_cards = filtered.sort_values(["best_probability", "winner_probability"], ascending=False)
    render_mobile_pick_cards(best_cards, limit=8)

    display_cols = [
        "time_lagos",
        "location",
        "match",
        "best_market",
        "best_pick",
        "best_probability",
        "best_strength",
        "winner_pick",
        "winner_probability",
        "winner_strength",
        "total_pick",
        "total_probability",
        "total_strength",
        "expected_total_points",
        "first_set_pick",
        "first_set_probability",
        "first_set_strength",
        "expected_first_set_points",
        "sets_pick",
        "sets_probability",
        "sets_strength",
        "expected_sets_played",
        "confidence",
        "upset_risk",
        "upset_risk_flags",
        "h2h_matches",
        "match_id",
    ]
    st.subheader("Full live prediction board")
    st.dataframe(format_prediction_table(filtered[display_cols]), use_container_width=True, height=560)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download live predictions CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="setka_live_predictions.csv",
            mime="text/csv",
        )
    with dl2:
        if st.button("Save snapshot in this session"):
            old = st.session_state.get("prediction_snapshots", pd.DataFrame())
            st.session_state["prediction_snapshots"] = pd.concat([old, pred_df], ignore_index=True)
            st.success(f"Saved {len(pred_df)} rows in this browser session.")

    if "prediction_snapshots" in st.session_state and not st.session_state["prediction_snapshots"].empty:
        with st.expander("Session prediction snapshots", expanded=False):
            st.dataframe(format_prediction_table(st.session_state["prediction_snapshots"].tail(100)), use_container_width=True)
            st.download_button(
                "Download session snapshots CSV",
                data=st.session_state["prediction_snapshots"].to_csv(index=False).encode("utf-8"),
                file_name="setka_prediction_snapshots.csv",
                mime="text/csv",
            )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.subheader("Strong winners")
        st.dataframe(
            format_prediction_table(pred_df.sort_values("winner_probability", ascending=False).head(7)[["time_lagos", "match", "winner_pick", "winner_probability", "upset_risk"]]),
            use_container_width=True,
        )
    with s2:
        st.subheader("Strong totals")
        st.dataframe(
            format_prediction_table(pred_df.sort_values("total_probability", ascending=False).head(7)[["time_lagos", "match", "total_pick", "total_probability", "expected_total_points"]]),
            use_container_width=True,
        )
    with s3:
        st.subheader("Strong 1st set")
        st.dataframe(
            format_prediction_table(pred_df.sort_values("first_set_probability", ascending=False).head(7)[["time_lagos", "match", "first_set_pick", "first_set_probability", "expected_first_set_points"]]),
            use_container_width=True,
        )
    with s4:
        st.subheader("Strong sets")
        st.dataframe(
            format_prediction_table(pred_df.sort_values("sets_probability", ascending=False).head(7)[["time_lagos", "match", "sets_pick", "sets_probability", "expected_sets_played"]]),
            use_container_width=True,
        )

    st.caption("Analytical estimates only — not guaranteed results or betting advice.")


elif page == "Live Match Center":
    st.title("📺 Live Match Center")
    st.markdown("Follow Setka matches currently being played: live score, set score, active player, and official viewing links.")

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Live dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.caption("Live match center")
        live_auto_refresh = st.checkbox("Auto-refresh live scores", value=False)
        live_refresh_seconds = st.selectbox("Live refresh every", [10, 15, 30, 60], index=1)
    if live_auto_refresh:
        enable_browser_auto_refresh(live_refresh_seconds)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Refresh live scores", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        st.link_button("Watch/Open official Setka site", OFFICIAL_SETKA_URL)
    with c3:
        st.caption("If the official stream is available in your region, use the official Setka link. We do not re-host or bypass live video restrictions.")

    try:
        live_df = load_official_live()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if live_df.empty:
        st.info("No live Setka matches returned right now. Showing upcoming matches below.")
        try:
            upcoming_now = load_official_nearest().head(12)
            st.dataframe(upcoming_now[[c for c in ["start_time_lagos", "location", "player1", "player2", "match_id"] if c in upcoming_now.columns]], use_container_width=True)
        except Exception:
            pass
        st.stop()

    live_df = live_df.copy()
    live_df["match"] = live_df["player1"] + " vs " + live_df["player2"]
    live_df["current_score"] = live_df["player1_score"].fillna(0).astype(int).astype(str) + " - " + live_df["player2_score"].fillna(0).astype(int).astype(str)
    live_df["watch_link"] = OFFICIAL_SETKA_URL
    live_df["schedule_link"] = live_df.apply(
        lambda r: f"https://tabletennis.setkacup.com/en/schedule?hall={int(r['location_id']) if pd.notna(r.get('location_id')) else ''}&period={int(r['day_period']) if pd.notna(r.get('day_period')) else ''}&date={r.get('start_date_lagos', '')}",
        axis=1,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Live feed rows", f"{len(live_df):,}")
    k2.metric("Currently live", f"{(live_df['status'] == 'Live').sum():,}" if "status" in live_df else "-")
    k3.metric("With set score", f"{live_df['set_scores'].astype(str).ne('').sum():,}" if "set_scores" in live_df else "-")
    k4.metric("Last refresh", pd.Timestamp.now(tz="Africa/Lagos").strftime("%H:%M:%S"))

    st.subheader("Live / just-finished feed")
    for _, r in live_df.sort_values(["start_time_lagos", "location"]).iterrows():
        st.markdown(
            f"""
<div class="pick-card">
  <div class="pick-title">🔴 {r.get('start_time_lagos', '')} • {r.get('match', '')}</div>
  <div class="pick-meta">{r.get('location', '')} • {r.get('tournament', '')} • Match ID {r.get('match_id', '')}</div>
  <div><b>Score:</b> {r.get('current_score', '')} &nbsp; | &nbsp; <b>Sets:</b> {r.get('set_scores', '') or 'In progress'} &nbsp; | &nbsp; <b>Active:</b> {r.get('active_player', '') or '-'}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    cols = ["start_time_lagos", "location", "status", "match", "current_score", "set_scores", "active_player", "total_points", "first_set_total", "match_id"]
    st.dataframe(live_df[[c for c in cols if c in live_df.columns]], use_container_width=True, height=420)

    st.subheader("Official viewing links")
    st.write("Use these buttons to open the official Setka site/schedule. Live video availability depends on Setka and your region.")
    link_rows = live_df[["match", "watch_link", "schedule_link"]].head(10)
    st.dataframe(link_rows, use_container_width=True)
    st.download_button(
        "Download live scores CSV",
        data=live_df.to_csv(index=False).encode("utf-8"),
        file_name="setka_live_scores.csv",
        mime="text/csv",
    )


elif page == "Owner Edge Engine":
    st.title("💎 Owner Edge Engine")
    st.markdown("A strict Setka-only control room: fewer picks, minimum value odds, no-bet discipline, and bankroll caps.")
    st.warning("This engine is built to reject weak matches. It cannot guarantee profit; the edge is in filtering and refusing bad prices.")

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Live dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    o1, o2, o3, o4, o5 = st.columns(5)
    with o1:
        owner_bankroll = st.number_input("Bankroll", min_value=0.0, value=10000.0, step=500.0, key="owner_bankroll")
    with o2:
        max_risk_pct = st.slider("Max stake %", 0.25, 5.0, 1.0, 0.25, key="owner_risk") / 100
    with o3:
        edge_buffer = st.slider("Required edge buffer", 1, 10, 3, 1, key="owner_edge_buffer") / 100
    with o4:
        owner_limit = st.slider("Matches to scan", 5, 50, 25, 5, key="owner_scan")
    with o5:
        if st.button("Refresh engine", type="primary"):
            st.cache_data.clear()
            st.rerun()

    l1, l2, l3 = st.columns(3)
    with l1:
        owner_first_line = st.number_input("1st set line", 10.5, 35.5, 18.5, 0.5, key="owner_first_line")
    with l2:
        owner_total_line = st.number_input("Total line", 30.5, 140.5, 75.5, 0.5, key="owner_total_line")
    with l3:
        owner_sets_line = st.selectbox("Sets line", [3.5, 4.5], index=0, key="owner_sets_line")

    try:
        upcoming = load_official_nearest().head(owner_limit)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    rows = []
    for _, match_row in upcoming.iterrows():
        if match_row.get("player1") and match_row.get("player2"):
            rows.append(prediction_pick_row(match_row, owner_first_line, owner_total_line, owner_sets_line))
    edge_df = apply_pick_strengths(pd.DataFrame(rows))
    if edge_df.empty:
        st.warning("No official Setka matches available right now.")
        st.stop()

    edge_df["edge_score"] = edge_df.apply(owner_edge_score, axis=1)
    edge_df["owner_decision"] = edge_df.apply(owner_decision, axis=1)
    edge_df["reason"] = edge_df.apply(owner_reason, axis=1)
    edge_df["fair_odds"] = edge_df["best_probability"].map(fair_decimal)
    edge_df["minimum_value_odds"] = edge_df["best_probability"].map(lambda p: minimum_value_odds(p, edge_buffer))
    edge_df["max_suggested_stake"] = edge_df.apply(
        lambda r: 0.0 if r["owner_decision"] == "NO BET" else min(owner_bankroll * max_risk_pct, owner_bankroll * max(0.0025, (r["edge_score"] - 0.58) / 18)),
        axis=1,
    )

    green = edge_df.loc[edge_df["owner_decision"] == "GREEN"].sort_values("edge_score", ascending=False)
    watch = edge_df.loc[edge_df["owner_decision"] == "WATCH"].sort_values("edge_score", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matches scanned", f"{len(edge_df):,}")
    k2.metric("GREEN", f"{len(green):,}")
    k3.metric("WATCH", f"{len(watch):,}")
    k4.metric("NO BET", f"{(edge_df['owner_decision'] == 'NO BET').sum():,}")

    st.subheader("GREEN picks — only if real odds are equal/above minimum value odds")
    cols = ["time_lagos", "location", "match", "best_market", "best_pick", "best_probability", "edge_score", "fair_odds", "minimum_value_odds", "max_suggested_stake", "confidence", "upset_risk", "h2h_matches", "reason"]
    green_display = green[cols].copy()
    for c in ["best_probability", "edge_score"]:
        green_display[c] = green_display[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for c in ["fair_odds", "minimum_value_odds", "max_suggested_stake"]:
        green_display[c] = green_display[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if green_display.empty:
        st.info("No GREEN picks right now. That is intentional: the owner engine protects bankroll by waiting.")
    else:
        st.dataframe(green_display, use_container_width=True, height=320)
        if st.button("Save GREEN picks to Strong Pick Tracker", key="save_owner_green"):
            count = save_strong_picks_to_tracker(green, "Owner Edge Engine")
            st.success(f"Saved {count} GREEN picks to Strong Pick Tracker.")

    with st.expander("Full decision board: WATCH and NO BET", expanded=False):
        full_cols = ["time_lagos", "location", "match", "best_market", "best_pick", "best_probability", "edge_score", "owner_decision", "minimum_value_odds", "confidence", "upset_risk", "upset_risk_flags", "reason"]
        full = edge_df[full_cols].sort_values(["owner_decision", "edge_score"], ascending=[True, False]).copy()
        for c in ["best_probability", "edge_score"]:
            full[c] = full[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        full["minimum_value_odds"] = full["minimum_value_odds"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(full, use_container_width=True, height=520)

    st.download_button(
        "Download owner edge report CSV",
        data=edge_df.to_csv(index=False).encode("utf-8"),
        file_name="owner_edge_report.csv",
        mime="text/csv",
    )

    st.caption("Owner rule: if bookmaker odds are below minimum value odds, it is a NO BET even if the model likes the pick.")


elif page == "Strong Pick Tracker":
    st.title("📌 Strong Pick Tracker")
    st.markdown("Automatically grades only GREEN/strong picks saved from the Trading Desk or Owner Edge Engine. It checks official Setka results for the saved pick dates — no weak picks mixed in.")
    if github_storage_enabled():
        st.success("Permanent GitHub storage is active. Strong picks/results will persist across redeploys until reset.")
    else:
        st.warning("GitHub permanent storage is not configured yet. Local storage may reset on Streamlit reboot/redeploy. Add GITHUB_STORAGE_TOKEN in Streamlit secrets for permanent saving.")

    tracker = load_strong_picks()
    session_tracker = st.session_state.get("strong_pick_tracker", pd.DataFrame())
    if not session_tracker.empty:
        tracker = save_strong_picks(session_tracker)

    upload = st.file_uploader("Optional: upload a previously downloaded strong-picks CSV", type=["csv"], key="strong_tracker_upload")
    if upload is not None:
        try:
            uploaded_tracker = pd.read_csv(upload)
            tracker = save_strong_picks(uploaded_tracker)
            st.success(f"Loaded and saved {len(uploaded_tracker):,} uploaded strong picks.")
        except Exception as exc:
            st.exception(exc)

    if tracker.empty:
        st.info("No strong picks saved yet. You can generate current GREEN picks now, or go to Setka Trading Desk / Owner Edge Engine and save GREEN picks.")
        if st.button("Generate current GREEN picks now", type="primary"):
            try:
                upcoming_now = load_official_nearest().head(30)
                rows_now = []
                for _, match_row in upcoming_now.iterrows():
                    if match_row.get("player1") and match_row.get("player2"):
                        rows_now.append(prediction_pick_row(match_row, 18.5, 75.5, 3.5))
                now_df = apply_pick_strengths(pd.DataFrame(rows_now))
                if not now_df.empty:
                    now_df["edge_score"] = now_df.apply(owner_edge_score, axis=1)
                    now_df["owner_decision"] = now_df.apply(owner_decision, axis=1)
                    now_df["reason"] = now_df.apply(owner_reason, axis=1)
                    now_df["fair_odds"] = now_df["best_probability"].map(fair_decimal)
                    now_df["minimum_value_odds"] = now_df["best_probability"].map(lambda p: minimum_value_odds(p, 0.03))
                    green_now = now_df.loc[now_df["owner_decision"] == "GREEN"].copy()
                    if green_now.empty:
                        st.warning("No GREEN picks right now. The tracker did not save weak picks.")
                    else:
                        save_strong_picks_to_tracker(green_now, "Strong Pick Tracker Auto-Generate")
                        st.success(f"Saved {len(green_now)} current GREEN picks. Reopening tracker...")
                        st.rerun()
                else:
                    st.warning("No upcoming matches returned right now.")
            except Exception as exc:
                st.exception(exc)
        st.stop()

    today_lagos = pd.Timestamp.now(tz="Africa/Lagos").date()
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    with c1:
        auto_check_dates = st.checkbox("Auto-check saved pick dates", value=True)
    with c2:
        track_date = st.date_input("Manual result date", value=today_lagos, key="strong_track_date")
    with c3:
        if st.button("Check now", type="primary"):
            st.cache_data.clear()
            st.rerun()

    track = tracker.copy()
    if "match_id" in track.columns:
        track["match_id"] = pd.to_numeric(track["match_id"], errors="coerce")

    dates_to_check = tracker_result_dates(track, track_date) if auto_check_dates else expand_setka_result_dates([str(track_date)])

    result_df = fetch_tracker_results(track, track_date) if auto_check_dates else sync_official_results_for_dates(dates_to_check)
    if result_df.empty:
        st.warning("No official results returned yet for the saved pick dates.")
        st.stop()

    if "match_id" in track.columns:
        track["match_id"] = pd.to_numeric(track["match_id"], errors="coerce")
        track = track.dropna(subset=["match_id"]).copy()
        track["match_id"] = track["match_id"].astype("int64")

    result_for_merge = result_df.copy()
    if "match_id" in result_for_merge.columns:
        result_for_merge["match_id"] = pd.to_numeric(result_for_merge["match_id"], errors="coerce")
        result_for_merge = result_for_merge.dropna(subset=["match_id"]).copy()
        result_for_merge["match_id"] = result_for_merge["match_id"].astype("int64")
    result_for_merge["actual_match"] = result_for_merge["player1"] + " vs " + result_for_merge["player2"]
    merged = track.merge(result_for_merge, on="match_id", how="left", suffixes=("_pick", "_result"))

    grades = merged.apply(grade_best_pick, axis=1, result_type="expand")
    merged["best_pick_grade"] = grades[0]
    merged["actual_best_market_result"] = grades[1]

    finished = merged.loc[merged["best_pick_grade"].isin(["✅", "❌"])]
    wins = int((finished["best_pick_grade"] == "✅").sum()) if not finished.empty else 0
    losses = int((finished["best_pick_grade"] == "❌").sum()) if not finished.empty else 0
    pending = int((merged["best_pick_grade"] == "Pending").sum())
    accuracy = wins / len(finished) if len(finished) else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tracked picks", f"{len(merged):,}")
    m2.metric("Won", f"{wins:,}")
    m3.metric("Lost", f"{losses:,}")
    m4.metric("Pending", f"{pending:,}")
    m5.metric("Strong-pick accuracy", format_percent(accuracy) if accuracy is not None else "-")
    matched_count = int(merged["status"].notna().sum()) if "status" in merged.columns else 0
    st.caption(f"Auto-checked official Setka result date(s): {', '.join(dates_to_check)} | Result rows found: {len(result_df):,} | Matched tracked picks: {matched_count:,}")
    if pending:
        st.info("Pending means the match has not finished yet, or the official result has not appeared in the Setka feed. The tracker now checks a wider ±3 day Setka window plus the live widget.")
        unmatched = merged.loc[merged.get("status", pd.Series(index=merged.index)).isna(), [c for c in ["match_id", "time_lagos", "match", "best_market", "best_pick", "date_lagos"] if c in merged.columns]]
        if not unmatched.empty:
            with st.expander("Debug: unmatched saved picks", expanded=False):
                st.dataframe(unmatched, use_container_width=True)

    st.subheader("Strong pick results")
    cols = [
        "saved_at_lagos", "tracker_source", "time_lagos", "location_pick", "match", "best_market", "best_pick",
        "best_probability", "minimum_value_odds", "stake_cap", "max_suggested_stake", "confidence", "upset_risk",
        "status", "winner", "set_scores", "total_points", "first_set_total", "sets_played",
        "actual_best_market_result", "best_pick_grade", "match_id"
    ]
    visible = merged[[c for c in cols if c in merged.columns]].copy()
    for c in ["best_probability"]:
        if c in visible:
            visible[c] = visible[c].map(lambda x: f"{x:.1%}" if pd.notna(x) and not isinstance(x, str) else x)
    st.dataframe(visible, use_container_width=True, height=520)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download tracked strong picks CSV", tracker.to_csv(index=False).encode("utf-8"), "strong_picks_saved.csv", "text/csv")
    with dl2:
        st.download_button("Download graded results CSV", merged.to_csv(index=False).encode("utf-8"), "strong_picks_graded.csv", "text/csv")

    st.caption(f"Strong picks are saved in: `{STRONG_PICKS_FILE}`. Synced official results are saved in: `{DAILY_RESULTS_FILE}`.")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("Reset strong picks permanently"):
            reset_strong_picks()
            st.session_state["strong_pick_tracker"] = pd.DataFrame()
            st.success("Strong Pick Tracker reset.")
            st.rerun()
    with col_reset2:
        if st.button("Reset saved daily results"):
            reset_daily_results()
            st.success("Saved daily results reset.")
            st.rerun()


elif page == "Results Checker":
    st.title("✅ Results Checker")
    st.markdown("Check official Setka results, live scores, set totals, and grade stored prediction snapshots.")

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Live dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.caption("Results refresh")
        results_auto_refresh = st.checkbox("Auto-refresh results page", value=False)
        results_refresh_seconds = st.selectbox("Results refresh every", [15, 30, 60, 120], index=1)
    if results_auto_refresh:
        enable_browser_auto_refresh(results_refresh_seconds)

    today_lagos = pd.Timestamp.now(tz="Africa/Lagos").date()
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        result_date = st.date_input("Setka date", value=today_lagos)
    with c2:
        period_label = st.selectbox("Day period", ["All", "Morning", "Evening", "Night"], index=0)
        period_map = {"All": None, "Morning": 1, "Evening": 2, "Night": 3}
    with c3:
        if st.button("Refresh results", type="primary"):
            st.cache_data.clear()
            st.rerun()

    try:
        result_df = load_official_results(str(result_date), period_map[period_label])
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if result_df.empty:
        st.warning("No official result rows returned for that date/period.")
        st.stop()

    # Persist official results whenever this page is opened/refreshed.
    saved_results = save_daily_results(result_df.assign(synced_at_lagos=pd.Timestamp.now(tz="Africa/Lagos").strftime("%Y-%m-%d %H:%M:%S")))

    rcols = st.columns(4)
    rcols[0].metric("Matches", f"{len(result_df):,}")
    rcols[1].metric("Finished", f"{(result_df['status'] == 'Finished').sum():,}")
    rcols[2].metric("Live", f"{(result_df['status'] == 'Live').sum():,}")
    rcols[3].metric("Scheduled", f"{(result_df['status'] == 'Scheduled').sum():,}")
    st.caption(f"Auto-saved official result rows in local app storage: {len(saved_results):,}")

    st.subheader("Official result feed")
    filter_text = st.text_input("Search player/location/match id", value="")
    view = result_df.copy()
    view["match"] = view["player1"] + " vs " + view["player2"]
    if filter_text:
        needle = filter_text.lower().strip()
        view = view.loc[
            view.apply(lambda r: needle in " ".join(str(v).lower() for v in r.values), axis=1)
        ]
    cols = ["start_time_lagos", "location", "status", "match", "player1_score", "player2_score", "winner", "set_scores", "total_points", "first_set_total", "match_id"]
    st.dataframe(view[[c for c in cols if c in view.columns]], use_container_width=True, height=430)

    with st.expander("Grade former prediction snapshot", expanded=True):
        st.caption("This grades the earlier predictions we generated in this chat for the first Setka night matches.")
        graded_rows = []
        for match_id, pred in FORMER_PREDICTIONS.items():
            row_match = result_df.loc[result_df["match_id"] == match_id]
            if row_match.empty:
                graded_rows.append({"match_id": match_id, **pred, "status": "Not found today"})
                continue
            row = row_match.iloc[0]
            actual_total = None
            if pd.notna(row.get("total_points")):
                actual_total = "Over" if row["total_points"] > 75.5 else "Under"
            actual_first = None
            if pd.notna(row.get("first_set_total")):
                actual_first = "Over" if row["first_set_total"] > 18.5 else "Under"
            graded_rows.append(
                {
                    "match_id": match_id,
                    "time_lagos": row.get("start_time_lagos"),
                    "match": f"{row.get('player1')} vs {row.get('player2')}",
                    "status": row.get("status"),
                    "score": f"{row.get('player1_score')} - {row.get('player2_score')}",
                    "sets": row.get("set_scores"),
                    "winner_pick": pred["winner_pick"],
                    "actual_winner": row.get("winner") or None,
                    "winner_grade": grade_pick(pred["winner_pick"], row.get("winner") or None),
                    "total_pick": pred["total_pick"],
                    "actual_total_pick": actual_total,
                    "total_points": row.get("total_points"),
                    "total_grade": grade_pick(pred["total_pick"], actual_total),
                    "first_set_pick": pred["first_set_pick"],
                    "actual_first_set_pick": actual_first,
                    "first_set_total": row.get("first_set_total"),
                    "first_set_grade": grade_pick(pred["first_set_pick"], actual_first),
                }
            )
        graded = pd.DataFrame(graded_rows)
        st.dataframe(graded, use_container_width=True)
        finished = graded.loc[graded["actual_winner"].notna()] if "actual_winner" in graded else pd.DataFrame()
        if not finished.empty:
            win_acc = (finished["winner_grade"] == "✅").mean()
            total_acc = (finished["total_grade"] == "✅").mean()
            first_acc = (finished["first_set_grade"] == "✅").mean()
            a1, a2, a3 = st.columns(3)
            a1.metric("Winner accuracy", format_percent(win_acc))
            a2.metric("Total accuracy", format_percent(total_acc))
            a3.metric("1st-set accuracy", format_percent(first_acc))

    with st.expander("Upload and grade your prediction CSV", expanded=False):
        st.caption("Upload a CSV downloaded from Live Predictions. It must include match_id, winner_pick, total_pick, and first_set_pick columns.")
        uploaded_predictions = st.file_uploader("Prediction CSV", type=["csv"], key="grade_prediction_csv")
        if uploaded_predictions is not None:
            try:
                user_preds = pd.read_csv(uploaded_predictions)
                needed = {"match_id", "winner_pick", "total_pick", "first_set_pick"}
                missing = needed - set(user_preds.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(sorted(missing))}")
                else:
                    result_for_merge = result_df.copy()
                    result_for_merge["actual_match"] = result_for_merge["player1"] + " vs " + result_for_merge["player2"]
                    if "start_time_lagos" in result_for_merge.columns:
                        result_for_merge = result_for_merge.rename(columns={"start_time_lagos": "actual_time_lagos"})
                    merged = user_preds.merge(
                        result_for_merge,
                        on="match_id",
                        how="left",
                        suffixes=("_pred", "_actual"),
                    )
                    merged["actual_total_pick"] = merged["total_points"].map(
                        lambda x: "Over" if pd.notna(x) and x > 75.5 else "Under" if pd.notna(x) else None
                    )
                    merged["actual_first_set_pick"] = merged["first_set_total"].map(
                        lambda x: "Over" if pd.notna(x) and x > 18.5 else "Under" if pd.notna(x) else None
                    )
                    merged["winner_grade"] = merged.apply(lambda r: grade_pick(r.get("winner_pick"), r.get("winner")), axis=1)
                    merged["total_grade"] = merged.apply(lambda r: grade_pick(r.get("total_pick"), r.get("actual_total_pick")), axis=1)
                    merged["first_set_grade"] = merged.apply(lambda r: grade_pick(r.get("first_set_pick"), r.get("actual_first_set_pick")), axis=1)
                    out_cols = [
                        "match_id",
                        "time_lagos",
                        "actual_time_lagos",
                        "location_pred",
                        "location_actual",
                        "match",
                        "actual_match",
                        "status",
                        "winner_pick",
                        "winner",
                        "winner_grade",
                        "total_pick",
                        "actual_total_pick",
                        "total_points",
                        "total_grade",
                        "first_set_pick",
                        "actual_first_set_pick",
                        "first_set_total",
                        "first_set_grade",
                    ]
                    st.dataframe(merged[[c for c in out_cols if c in merged.columns]], use_container_width=True)
                    finished_upload = merged.loc[merged["winner"].notna()]
                    if not finished_upload.empty:
                        u1, u2, u3 = st.columns(3)
                        u1.metric("Winner accuracy", format_percent((finished_upload["winner_grade"] == "✅").mean()))
                        u2.metric("Total accuracy", format_percent((finished_upload["total_grade"] == "✅").mean()))
                        u3.metric("1st-set accuracy", format_percent((finished_upload["first_set_grade"] == "✅").mean()))
            except Exception as exc:
                st.exception(exc)


elif page == "Match Predictor": 
    st.title("🏓 Match Predictor")
    st.markdown(
        "Estimate match winner, expected total points, and **first set Over/Under 18.5** from the Setka history."
    )

    c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.1, 1.1, 1.0])
    with c1:
        player_a = st.selectbox("Player A", players_by_elo, index=0)
    with c2:
        default_b = 1 if len(players_by_elo) > 1 else 0
        player_b = st.selectbox("Player B", players_by_elo, index=default_b)
    with c3:
        first_set_line = st.number_input(
            "1st set line", min_value=10.5, max_value=35.5, value=18.5, step=0.5
        )
    with c4:
        total_points_line = st.number_input(
            "Total points line", min_value=30.5, max_value=140.5, value=75.5, step=0.5
        )
    with c5:
        sets_line = st.selectbox("Sets line", [3.5, 4.5], index=0)

    if player_a == player_b:
        st.error("Choose two different players.")
        st.stop()

    pred = predict_match(
        player_a,
        player_b,
        player_stats,
        matches,
        global_stats,
        first_set_line=first_set_line,
        total_points_line=total_points_line,
        sets_line=sets_line,
    )

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predicted winner", pred["predicted_winner"])
    m2.metric(f"{player_a} win chance", format_percent(pred["player_a_win_probability"]))
    m3.metric(
        f"1st set Over {first_set_line}",
        format_percent(pred["first_set_over_probability"]),
    )
    m4.metric(
        f"Sets Over {sets_line}",
        format_percent(pred["sets_over_probability"]),
    )
    m5.metric("Upset risk", pred.get("upset_risk", "-"), help="Flags weak/conflicting winner picks.")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.plotly_chart(probability_bar(pred), use_container_width=True)
    with r2:
        st.plotly_chart(
            over_under_bar(
                f"1st set O/U {first_set_line}",
                pred["first_set_over_probability"],
                pred["first_set_under_probability"],
            ),
            use_container_width=True,
        )
    with r3:
        st.plotly_chart(
            over_under_bar(
                f"Sets O/U {sets_line}",
                pred["sets_over_probability"],
                pred["sets_under_probability"],
            ),
            use_container_width=True,
        )

    line_cols = st.columns(5)
    line_cols[0].metric("Expected 1st-set points", format_number(pred["expected_first_set_points"], 2))
    line_cols[1].metric("Expected total points", format_number(pred["expected_total_points"], 2))
    line_cols[2].metric("Expected sets", format_number(pred["expected_sets_played"], 2))
    line_cols[3].metric(
        f"Total Over {total_points_line}",
        format_percent(pred["total_points_over_probability"]),
    )
    line_cols[4].metric(
        f"Total Under {total_points_line}",
        format_percent(pred["total_points_under_probability"]),
    )

    total_pick = "Over" if pred["total_points_over_probability"] >= pred["total_points_under_probability"] else "Under"
    total_prob = max(pred["total_points_over_probability"], pred["total_points_under_probability"])
    first_pick = "Over" if pred["first_set_over_probability"] >= pred["first_set_under_probability"] else "Under"
    first_prob = max(pred["first_set_over_probability"], pred["first_set_under_probability"])
    sets_pick = "Over" if pred["sets_over_probability"] >= pred["sets_under_probability"] else "Under"
    sets_prob = max(pred["sets_over_probability"], pred["sets_under_probability"])
    winner_prob = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
    st.info(
        f"Pick strength → Winner: {pick_strength(winner_prob, pred['confidence'], pred.get('upset_risk'))} | "
        f"Total: {total_pick} {pick_strength(total_prob, pred['confidence'], pred.get('upset_risk'))} | "
        f"1st set: {first_pick} {pick_strength(first_prob, pred['confidence'], pred.get('upset_risk'))} | "
        f"Sets: {sets_pick} {pick_strength(sets_prob, pred['confidence'], pred.get('upset_risk'))}"
    )

    with st.expander("Manual odds value check", expanded=False):
        st.caption("Optional: compare your model probability with bookmaker decimal odds. Positive edge means model probability is higher than implied probability.")
        odds_market = st.selectbox("Market", ["Winner", "Total", "1st Set", "Sets"], key="manual_odds_market")
        if odds_market == "Winner":
            model_probability = winner_prob
            model_pick = pred["predicted_winner"]
        elif odds_market == "Total":
            model_probability = total_prob
            model_pick = total_pick
        elif odds_market == "1st Set":
            model_probability = first_prob
            model_pick = first_pick
        else:
            model_probability = sets_prob
            model_pick = f"{sets_pick} {sets_line} sets"
        decimal_odds = st.number_input("Decimal odds", min_value=1.01, max_value=50.0, value=1.80, step=0.01)
        implied_probability = 1 / decimal_odds
        edge = model_probability - implied_probability
        ev = (model_probability * decimal_odds) - 1
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Model pick", model_pick)
        o2.metric("Model probability", format_percent(model_probability))
        o3.metric("Implied probability", format_percent(implied_probability))
        o4.metric("Estimated edge", format_percent(edge), delta=format_percent(ev))

    with st.expander("Why this prediction?", expanded=True):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Elo difference", f"{pred['elo_diff']:+.1f}")
        d2.metric("Elo-only chance", format_percent(pred["elo_probability"]))
        d3.metric("H2H matches", f"{pred['h2h_matches']}")
        d4.metric(
            f"{player_a} H2H wins",
            f"{pred['h2h_player_a_wins']} / {pred['h2h_matches']}",
        )
        st.dataframe(comparison_table(player_stats, player_a, player_b), use_container_width=True)
        st.caption(
            "This is an analytical estimate, not a guarantee or financial advice. Use your own judgement."
        )

    st.subheader("Recent head-to-head matches")
    if pred["h2h_table"].empty:
        st.info("No direct head-to-head matches found in the uploaded history.")
    else:
        st.dataframe(h2h_display_table(pred["h2h_table"]), use_container_width=True)


elif page == "Accuracy Lab":
    st.title("🎯 Accuracy Lab")
    st.markdown("Time-split backtesting for the Setka rule model. This helps us trust only the best filters instead of every pick.")

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        test_rows = st.selectbox("Holdout matches", [250, 500, 750, 1000, 1500], index=2)
    with b2:
        bt_first_line = st.number_input("1st set line", 10.5, 35.5, 18.5, 0.5, key="bt_first")
    with b3:
        bt_total_line = st.number_input("Total line", 30.5, 140.5, 75.5, 0.5, key="bt_total")
    with b4:
        bt_sets_line = st.selectbox("Sets line", [3.5, 4.5], index=0, key="bt_sets")
    with b5:
        run_bt = st.button("Run backtest", type="primary")

    if not run_bt and "backtest_df" not in st.session_state:
        st.info("Click Run backtest to test recent historical Setka matches without using those match results for training.")
        st.stop()

    if run_bt:
        bt_df, bt_metrics = run_backtest_cached(test_rows, bt_first_line, bt_total_line, bt_sets_line)
        st.session_state["backtest_df"] = bt_df
        st.session_state["backtest_metrics"] = bt_metrics
    else:
        bt_df = st.session_state["backtest_df"]
        bt_metrics = st.session_state["backtest_metrics"]

    if bt_df.empty:
        st.warning("No backtest rows generated.")
        st.stop()

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Backtest rows", f"{bt_metrics.get('rows', 0):,}")
    a2.metric("Winner accuracy", format_percent(bt_metrics.get("winner_accuracy")))
    a3.metric("Total accuracy", format_percent(bt_metrics.get("total_accuracy")))
    a4.metric("1st-set accuracy", format_percent(bt_metrics.get("first_set_accuracy")))
    a5.metric("Sets O/U accuracy", format_percent(bt_metrics.get("sets_accuracy")))

    st.subheader("Accuracy by probability threshold")
    th = threshold_table(bt_df)
    st.dataframe(
        th.assign(
            min_probability=th["min_probability"].map(lambda x: f"{x:.0%}"),
            accuracy=th["accuracy"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        ),
        use_container_width=True,
    )

    st.subheader("Recommended live filters from this test")
    recommendations = []
    for market, prob_col, correct_col in [
        ("Winner", "winner_probability", "winner_correct"),
        ("Total", "total_probability", "total_correct"),
        ("1st Set", "first_set_probability", "first_set_correct"),
        ("Sets", "sets_probability", "sets_correct"),
    ]:
        best = None
        for threshold in [0.55, 0.58, 0.60, 0.65, 0.70]:
            subset = bt_df.loc[bt_df[prob_col] >= threshold]
            if len(subset) < max(20, len(bt_df) * 0.03):
                continue
            acc = subset[correct_col].mean()
            candidate = (acc, threshold, len(subset))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            recommendations.append({"market": market, "recommended_min_probability": best[1], "backtest_accuracy": best[0], "sample_picks": best[2]})
    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        st.dataframe(
            rec_df.assign(
                recommended_min_probability=rec_df["recommended_min_probability"].map(lambda x: f"{x:.0%}"),
                backtest_accuracy=rec_df["backtest_accuracy"].map(lambda x: f"{x:.1%}"),
            ),
            use_container_width=True,
        )

    st.subheader("Recent backtest rows")
    view = bt_df.sort_values("date_time", ascending=False).head(250).copy()
    for col in ["winner_probability", "total_probability", "first_set_probability"]:
        view[col] = view[col].map(lambda x: f"{x:.1%}")
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("Download backtest CSV", bt_df.to_csv(index=False).encode("utf-8"), "setka_backtest.csv", "text/csv")


elif page == "Smart Stake Calc":
    st.title("🧮 Smart Stake Calculator")
    st.markdown("Calculate payout, implied probability, model edge, expected value, and safer fractional-Kelly stake sizing.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bankroll = st.number_input("Bankroll", min_value=0.0, value=10000.0, step=500.0)
    with c2:
        stake = st.number_input("Planned stake", min_value=0.0, value=500.0, step=50.0)
    with c3:
        odds = st.number_input("Decimal odds", min_value=1.01, value=1.80, step=0.01)
    with c4:
        model_prob = st.slider("Your/model probability", 1, 99, 55) / 100

    implied = implied_probability(odds)
    payout = stake * odds
    profit = payout - stake
    edge = model_prob - implied
    ev = (model_prob * profit) - ((1 - model_prob) * stake)
    ev_pct = ev / stake if stake else 0
    kelly = kelly_fraction(model_prob, odds)
    safer_kelly = kelly * 0.25
    suggested = min(bankroll * safer_kelly, bankroll * 0.05) if bankroll else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Potential payout", f"{payout:,.2f}")
    s2.metric("Potential profit", f"{profit:,.2f}")
    s3.metric("Implied probability", format_percent(implied))
    s4.metric("Model edge", format_percent(edge))

    e1, e2, e3 = st.columns(3)
    e1.metric("Expected value", f"{ev:,.2f}", delta=format_percent(ev_pct))
    e2.metric("Full Kelly", format_percent(kelly))
    e3.metric("Safer 1/4 Kelly stake", f"{suggested:,.2f}")

    if edge <= 0:
        st.warning("No value detected: model probability is not higher than implied probability.")
    elif suggested < stake:
        st.info("Value detected, but your planned stake is above the safer 1/4 Kelly suggestion.")
    else:
        st.success("Value detected and planned stake is within safer sizing range.")

    st.caption("Kelly sizing is only a mathematical tool. Use small stakes, avoid chasing losses, and manage risk.")


elif page == "Bet Slip Tools":
    st.title("🎟️ Bet Slip Tools")
    st.markdown("Paste decimal odds to calculate combined odds, payout, and slip risk. This is the first version of merge/split-style bet tools.")

    b1, b2 = st.columns([1.4, 1])
    with b1:
        odds_text = st.text_area("Paste decimal odds", value="1.45\n1.80\n2.10", height=180, help="Separate odds with spaces, commas, lines, /, |, or ;")
    with b2:
        slip_stake = st.number_input("Slip stake", min_value=0.0, value=1000.0, step=100.0)
        target_parts = st.slider("Split into slips", 1, 10, 2)

    odds_list = parse_decimal_odds(odds_text)
    if not odds_list:
        st.warning("Enter decimal odds above 1.00.")
        st.stop()

    combined = combined_decimal_odds(odds_list)
    implied_combo = implied_probability(combined)
    payout = slip_stake * combined
    profit = payout - slip_stake
    risk_label = "Low" if len(odds_list) <= 2 and combined < 4 else "Medium" if len(odds_list) <= 4 and combined < 12 else "High"

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Selections", f"{len(odds_list)}")
    q2.metric("Combined odds", f"{combined:.2f}")
    q3.metric("Potential payout", f"{payout:,.2f}")
    q4.metric("Risk level", risk_label)

    st.subheader("Slip breakdown")
    slip_df = pd.DataFrame({"selection": range(1, len(odds_list) + 1), "decimal_odds": odds_list})
    slip_df["implied_probability"] = slip_df["decimal_odds"].map(implied_probability)
    st.dataframe(slip_df, use_container_width=True)

    split_stake = slip_stake / target_parts if target_parts else slip_stake
    st.subheader("Simple split staking")
    st.write(f"If you split **{slip_stake:,.2f}** into **{target_parts}** slips, each slip stake is about **{split_stake:,.2f}**.")
    st.info("Next upgrade can split selections into multiple lower-risk combinations automatically.")

    st.download_button(
        "Download slip CSV",
        data=slip_df.to_csv(index=False).encode("utf-8"),
        file_name="bet_slip_odds.csv",
        mime="text/csv",
    )


elif page == "ML Lab":
    st.title("🤖 ML Lab — scikit-learn / XGBoost")
    st.markdown(
        "Train machine-learning models for match winner, match total points, and first-set Over/Under 18.5."
    )

    if ML_IMPORT_ERROR is not None:
        st.error(f"ML dependencies could not be imported: {ML_IMPORT_ERROR}")
        st.info("Run `pip install -r requirements.txt` and restart the app.")
        st.stop()

    st.info(
        "The ML pipeline uses chronological pre-match features to reduce future leakage: rolling Elo, player form, point totals, first-set trends, and H2H history."
    )

    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    with c1:
        algorithm = st.selectbox(
            "Algorithm",
            ["auto", "xgboost", "sklearn"],
            help="auto uses XGBoost when installed, otherwise scikit-learn HistGradientBoosting.",
        )
    with c2:
        row_cap_label = st.selectbox(
            "Training size",
            ["Quick: latest 50k rows", "Balanced: latest 120k rows", "Full: all rows"],
            index=1,
            help="Rows are orientation rows; each match creates two rows. Full data is most complete but slower.",
        )
        row_cap_map = {
            "Quick: latest 50k rows": 50_000,
            "Balanced: latest 120k rows": 120_000,
            "Full: all rows": None,
        }
        max_training_rows = row_cap_map[row_cap_label]
    with c3:
        st.metric("Available ML rows", f"{len(matches) * 2:,}")

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        train_clicked = st.button("Train model", type="primary")
    with b2:
        load_clicked = st.button("Load saved model")
    with b3:
        if MODEL_BUNDLE_PATH.exists():
            st.caption(f"Saved bundle found: `{MODEL_BUNDLE_PATH}`")
        else:
            st.caption("No saved model bundle yet. Train one here or run `python scripts/train_models.py`.")

    if load_clicked:
        if MODEL_BUNDLE_PATH.exists():
            st.session_state["ml_bundle"] = load_model_bundle(MODEL_BUNDLE_PATH)
            st.success("Loaded saved ML bundle.")
        else:
            st.warning("No saved model bundle found yet.")

    if train_clicked:
        try:
            st.session_state["ml_bundle"] = train_ml_cached(algorithm, max_training_rows)
            st.success("ML training complete.")
        except Exception as exc:
            st.exception(exc)
            st.stop()

    bundle = st.session_state.get("ml_bundle")
    if not bundle:
        st.stop()

    st.divider()
    meta_cols = st.columns(5)
    meta_cols[0].metric("Algorithm", bundle["algorithm"])
    meta_cols[1].metric("Rows used", f"{bundle['rows_used_for_training']:,}")
    meta_cols[2].metric("Train rows", f"{bundle['train_rows']:,}")
    meta_cols[3].metric("Test rows", f"{bundle['test_rows']:,}")
    meta_cols[4].metric("Models", "4")

    st.subheader("Holdout metrics")
    st.dataframe(metrics_table(bundle), use_container_width=True)

    with st.expander("Save / reuse this model", expanded=False):
        st.write("Save the trained bundle locally so the app can load it next time.")
        if st.button("Save model bundle to models/setka_ml_bundle.joblib"):
            saved_path = save_model_bundle(bundle, MODEL_BUNDLE_PATH)
            st.success(f"Saved: {saved_path}")
        st.code("python scripts/train_models.py --algorithm auto --output models/setka_ml_bundle.joblib", language="bash")

    st.subheader("ML prediction")
    p1, p2, p3, p4 = st.columns([2.2, 2.2, 1.25, 1.25])
    with p1:
        ml_player_a = st.selectbox("Player A", players_by_elo, index=0, key="ml_a")
    with p2:
        ml_player_b = st.selectbox("Player B", players_by_elo, index=1 if len(players_by_elo) > 1 else 0, key="ml_b")
    with p3:
        ml_first_line = st.number_input("1st set line", min_value=10.5, max_value=35.5, value=18.5, step=0.5, key="ml_first_line")
    with p4:
        ml_total_line = st.number_input("Total points line", min_value=30.5, max_value=140.5, value=75.5, step=0.5, key="ml_total_line")

    if ml_player_a == ml_player_b:
        st.error("Choose two different players.")
        st.stop()

    ml_pred = predict_with_bundle(
        bundle,
        ml_player_a,
        ml_player_b,
        first_set_line=ml_first_line,
        total_points_line=ml_total_line,
    )
    rule_pred = predict_match(
        ml_player_a,
        ml_player_b,
        player_stats,
        matches,
        global_stats,
        first_set_line=ml_first_line,
        total_points_line=ml_total_line,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ML predicted winner", ml_pred["predicted_winner"])
    m2.metric(f"{ml_player_a} ML win chance", format_percent(ml_pred["player_a_win_probability"]))
    m3.metric(f"ML 1st set Over {ml_first_line}", format_percent(ml_pred["first_set_over_probability"]))
    m4.metric("ML expected total", format_number(ml_pred["expected_total_points"], 2))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(probability_bar(ml_pred), use_container_width=True)
    with c2:
        st.plotly_chart(
            over_under_bar(
                f"ML first set O/U {ml_first_line}",
                ml_pred["first_set_over_probability"],
                ml_pred["first_set_under_probability"],
            ),
            use_container_width=True,
        )

    compare_df = pd.DataFrame(
        [
            {
                "model": "ML",
                "predicted_winner": ml_pred["predicted_winner"],
                f"{ml_player_a}_win_probability": ml_pred["player_a_win_probability"],
                "expected_total_points": ml_pred["expected_total_points"],
                "total_over_probability": ml_pred["total_points_over_probability"],
                "expected_first_set_points": ml_pred["expected_first_set_points"],
                "first_set_over_probability": ml_pred["first_set_over_probability"],
            },
            {
                "model": "Rule blend",
                "predicted_winner": rule_pred["predicted_winner"],
                f"{ml_player_a}_win_probability": rule_pred["player_a_win_probability"],
                "expected_total_points": rule_pred["expected_total_points"],
                "total_over_probability": rule_pred["total_points_over_probability"],
                "expected_first_set_points": rule_pred["expected_first_set_points"],
                "first_set_over_probability": rule_pred["first_set_over_probability"],
            },
        ]
    )
    st.subheader("ML vs rule-blend comparison")
    st.dataframe(compare_df, use_container_width=True)
    st.caption("ML estimates are based on historical patterns only. They are not betting advice or guaranteed outcomes.")


elif page == "Data Sources":
    st.title("🧭 Data Sources & Research Registry")
    st.markdown(
        "A structured registry of the live-score, odds, table-tennis, ML, training, GitHub, and research resources you listed."
    )

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Source registry dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.stop()

    st.warning(
        "Compliance note: the app does not blindly scrape websites. Use official APIs, licensed feeds, permitted exports, or manual imports."
    )

    s1, s2 = st.columns([1, 2])
    with s1:
        selected_category = st.selectbox("Category", ["All"] + source_categories())
    with s2:
        st.caption(
            "Status tells you whether the source is already wired into the app, available as a scaffold, or kept as a research/manual-reference link."
        )

    st.subheader("Summary")
    st.dataframe(summary_by_category(), use_container_width=True)

    st.subheader("Sources")
    source_df = registry_dataframe(selected_category)
    st.dataframe(
        source_df,
        use_container_width=True,
        height=620,
        column_config={
            "url": st.column_config.LinkColumn("url"),
        },
    )

    with st.expander("Secrets for API integrations", expanded=False):
        st.code(
            """THE_ODDS_API_KEY="..."
PINNACLE_USERNAME="..."
PINNACLE_PASSWORD="..."
BETFAIR_APP_KEY="..."
BETFAIR_SESSION_TOKEN="...""",
            language="bash",
        )

    with st.expander("Recommended next build steps", expanded=False):
        st.markdown(
            """
1. Add your GitHub repo URL so the project can be pushed.
2. Add The Odds API key and discover the exact table-tennis sport key available to your account.
3. If you have Pinnacle/Betfair access, wire the approved endpoints into `src/external_clients.py`.
4. Add a canonical player-name mapping table for matching odds/live-score names to Setka CSV names.
5. Backtest predictions against historical odds before relying on any edge calculation.
            """
        )


elif page == "Leaderboard":
    st.title("🏆 Leaderboard")
    st.markdown("Search and rank players by Elo, matches, win rate, form, and point-total profile.")

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("Search player", placeholder="Type part of a player name...")
    with fc2:
        min_matches = st.number_input("Minimum history matches", min_value=0, value=0, step=10)
    with fc3:
        sort_by = st.selectbox(
            "Sort by",
            ["elo", "matches", "win_rate", "recent_win_rate", "avg_total_points", "first_set_over_18_5_rate"],
        )

    df = player_stats.copy()
    if search:
        df = df[df["player"].str.contains(search, case=False, na=False)]
    df = df[df["matches"] >= min_matches]
    df = df.sort_values(sort_by, ascending=False)

    show_cols = [
        "player",
        "elo",
        "matches",
        "wins",
        "losses",
        "win_rate",
        "recent_win_rate",
        "avg_total_points",
        "avg_first_set_total",
        "first_set_over_18_5_rate",
        "last_played",
    ]
    st.dataframe(df[show_cols], use_container_width=True, height=600)
    st.download_button(
        "Download filtered leaderboard CSV",
        data=df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="setka_filtered_leaderboard.csv",
        mime="text/csv",
    )


elif page == "Player Explorer":
    st.title("🔎 Player Explorer")
    player = st.selectbox("Choose player", players_alpha)
    row = player_stats.loc[player_stats["player"] == player].iloc[0]
    log = player_log.loc[player_log["player"] == player].copy().sort_values("date_time")

    st.subheader(player)
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Elo", f"{row['elo']:.1f}")
    p2.metric("Matches", f"{int(row['matches']):,}")
    p3.metric("Win rate", format_percent(row["win_rate"]))
    p4.metric("Recent win rate", format_percent(row["recent_win_rate"]))
    p5.metric("1st set O18.5 rate", format_percent(row["first_set_over_18_5_rate"]))

    if log.empty:
        st.info("No match history for this player in the uploaded match file.")
        st.stop()

    chart_df = log[["date_time", "won", "total_points", "first_set_total", "point_diff"]].copy()
    chart_df["rolling_win_rate_20"] = chart_df["won"].rolling(20, min_periods=3).mean()
    chart_df["match_number"] = range(1, len(chart_df) + 1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            chart_df,
            x="date_time",
            y="rolling_win_rate_20",
            title="Rolling win rate / last 20 matches",
            labels={"rolling_win_rate_20": "Rolling win rate", "date_time": "Date"},
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            chart_df,
            x="total_points",
            nbins=35,
            title="Match total points distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(
            chart_df,
            x="first_set_total",
            nbins=25,
            title="First-set points distribution",
        )
        fig.add_vline(x=18.5, line_dash="dash", line_color="#F97316")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        recent_opponents = (
            log.groupby("opponent")
            .agg(matches=("won", "size"), wins=("won", "sum"), avg_total_points=("total_points", "mean"))
            .reset_index()
        )
        recent_opponents["win_rate"] = recent_opponents["wins"] / recent_opponents["matches"]
        recent_opponents = recent_opponents.sort_values("matches", ascending=False).head(15)
        fig = px.bar(
            recent_opponents,
            x="matches",
            y="opponent",
            orientation="h",
            title="Most common opponents",
            hover_data=["wins", "win_rate", "avg_total_points"],
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest matches")
    st.dataframe(player_latest_table(log), use_container_width=True)


elif page == "Head-to-Head":
    st.title("⚔️ Head-to-Head")
    c1, c2 = st.columns(2)
    with c1:
        player_a = st.selectbox("Player A", players_alpha, index=0, key="h2h_a")
    with c2:
        player_b = st.selectbox("Player B", players_alpha, index=1 if len(players_alpha) > 1 else 0, key="h2h_b")

    if player_a == player_b:
        st.error("Choose two different players.")
        st.stop()

    summary, h2h_rows = get_head_to_head(matches, player_a, player_b)

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("H2H matches", f"{summary['matches']}")
    h2.metric(f"{player_a} wins", f"{summary['player_a_wins']}")
    h3.metric(f"{player_b} wins", f"{summary['player_b_wins']}")
    h4.metric("Avg total points", format_number(summary["avg_total_points"], 2))
    h5.metric("1st set O18.5", format_percent(summary["first_set_over_18_5_rate"]))

    if h2h_rows.empty:
        st.info("No direct H2H matches found for these players.")
    else:
        chart = h2h_rows.sort_values("date_time").copy()
        chart["player_a_cum_wins"] = chart["selected_player_won"].cumsum()
        chart["match_no"] = range(1, len(chart) + 1)
        fig = px.line(
            chart,
            x="match_no",
            y="player_a_cum_wins",
            title=f"Cumulative H2H wins for {player_a}",
            labels={"match_no": "H2H match number", "player_a_cum_wins": "Cumulative wins"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(h2h_display_table(h2h_rows, limit=100), use_container_width=True)


elif page == "Live Odds":
    st.title("📡 Live Odds + Setka Links")
    st.markdown(
        "Connect The Odds API when you add an API key, and use the official Setka Cup site for schedules/results."
    )

    if ODDS_IMPORT_ERROR is not None:
        st.error(f"Odds/API dependencies could not be imported: {ODDS_IMPORT_ERROR}")
        st.info("Run `pip install -r requirements.txt` and restart the app.")
        st.stop()

    st.subheader("Official Setka Cup")
    oc1, oc2, oc3 = st.columns([1.3, 1, 2])
    with oc1:
        st.link_button("Open Setka Cup official site", OFFICIAL_SETKA_URL)
    with oc2:
        check_site = st.button("Check site status")
    with oc3:
        st.caption(
            "The official Setka website can be dynamic. This app checks availability; plug in an official feed/API here if you have one."
        )
    if check_site:
        status = fetch_official_site_status()
        st.json(status_as_dict(status))

    st.divider()
    st.subheader("The Odds API")

    secret_key = None
    try:
        secret_key = st.secrets.get("THE_ODDS_API_KEY")
    except Exception:
        secret_key = None
    env_key = os.getenv("THE_ODDS_API_KEY")
    entered_key = st.text_input(
        "The Odds API key for this session",
        type="password",
        value="",
        help="Leave blank to use THE_ODDS_API_KEY from environment or Streamlit secrets.",
    )
    api_key = entered_key or env_key or secret_key

    if not api_key:
        st.warning("No Odds API key found yet.")
        st.write("Add it locally as an environment variable:")
        st.code("export THE_ODDS_API_KEY='your_api_key_here'\nstreamlit run app.py", language="bash")
        st.write("Or in Streamlit Cloud secrets:")
        st.code('THE_ODDS_API_KEY = "your_api_key_here"', language="toml")
        st.info("The code is already built; live odds will work after you add the key.")
    else:
        st.success("API key detected for this session.")

    with st.expander("Discover sport keys", expanded=False):
        all_sports = st.checkbox("Include inactive sports", value=False)
        if st.button("List sports from The Odds API"):
            if not api_key:
                st.error("Add an API key first.")
            else:
                try:
                    sports_df, quota = list_sports(api_key, all_sports=all_sports)
                    st.caption(f"Quota headers: {quota}")
                    st.dataframe(sports_df, use_container_width=True, height=420)
                except OddsAPIError as exc:
                    st.error(str(exc))

    st.markdown("### Fetch odds")
    f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.4, 1])
    with f1:
        sport_key = st.text_input(
            "Sport key",
            value="table_tennis",
            help="Use 'List sports' to find the exact key available to your account.",
        )
    with f2:
        regions = st.text_input("Regions", value="eu,uk,us")
    with f3:
        markets = st.text_input("Markets", value="h2h,totals")
    with f4:
        odds_format = st.selectbox("Odds format", ["decimal", "american"], index=0)

    if st.button("Fetch odds"):
        if not api_key:
            st.error("Add an API key first.")
        else:
            try:
                events, quota = fetch_odds(
                    api_key,
                    sport_key=sport_key,
                    regions=regions,
                    markets=markets,
                    odds_format=odds_format,
                )
                flat = normalize_odds_events(events)
                flat = add_implied_probabilities(flat, odds_format=odds_format)
                st.caption(f"Quota headers: {quota}")
                st.metric("Events returned", f"{len(events):,}")
                if flat.empty:
                    st.info("No odds rows returned for this sport/region/market combination.")
                else:
                    st.dataframe(flat, use_container_width=True, height=520)
                    st.download_button(
                        "Download odds CSV",
                        data=flat.to_csv(index=False).encode("utf-8"),
                        file_name=f"odds_{sport_key}.csv",
                        mime="text/csv",
                    )
            except OddsAPIError as exc:
                st.error(str(exc))
                st.info(
                    "If the sport key is invalid or unavailable, use 'List sports' to find the exact key. Some accounts/plans may not include table tennis or Setka markets."
                )

    with st.expander("How to compare odds with app predictions", expanded=False):
        st.markdown(
            """
1. Fetch odds for the correct table-tennis sport key and markets.
2. Match the event player names to the names in this dataset.
3. Use Match Predictor or ML Lab to estimate probabilities.
4. Compare model probability to bookmaker implied probability. For decimal odds, implied probability is `1 / price` before bookmaker margin removal.
            """
        )


elif page == "Data Health":
    st.title("🧪 Data Health")
    st.markdown("Quick checks on the uploaded files and parsed scoring data.")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Match rows", f"{len(matches):,}")
    g2.metric("Players", f"{player_stats['player'].nunique():,}")
    g3.metric("Avg total points", format_number(global_stats["total_points_mean"], 2))
    g4.metric("Avg 1st-set points", format_number(global_stats["first_set_mean"], 2))

    c1, c2 = st.columns(2)
    with c1:
        set_counts = matches["sets_played"].value_counts().sort_index().reset_index()
        set_counts.columns = ["sets_played", "matches"]
        fig = px.bar(set_counts, x="sets_played", y="matches", title="Matches by number of sets")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(matches, x="first_set_total", nbins=35, title="First-set total distribution")
        fig.add_vline(x=18.5, line_dash="dash", line_color="#F97316")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        top_comp = matches["competition"].value_counts().head(20).reset_index()
        top_comp.columns = ["competition", "matches"]
        fig = px.bar(top_comp, x="matches", y="competition", orientation="h", title="Top competitions")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        missing = matches.isna().sum().reset_index()
        missing.columns = ["column", "missing_values"]
        st.dataframe(missing, use_container_width=True)

    st.subheader("Raw match sample")
    sample_cols = [
        "date_time",
        "competition",
        "player1",
        "player2",
        "winner",
        "set_scores",
        "total_points",
        "first_set_total",
        "sets_played",
    ]
    st.dataframe(matches[sample_cols].head(100), use_container_width=True)
