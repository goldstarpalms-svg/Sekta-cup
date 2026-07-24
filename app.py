from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    from src.setka_live import OFFICIAL_SETKA_URL, fetch_official_site_status, status_as_dict
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and preparing Setka data...")
def load_app_context() -> dict:
    matches, leaderboard = load_raw_data()
    return build_context(matches, leaderboard)


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


with st.sidebar:
    st.title("🏓 Setka Predictor")
    st.caption("Prediction dashboard from uploaded Setka match history + Elo leaderboard.")
    st.divider()

    page = st.radio(
        "Go to",
        [
            "Match Predictor",
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


if page == "Match Predictor":
    st.title("🏓 Match Predictor")
    st.markdown(
        "Estimate match winner, expected total points, and **first set Over/Under 18.5** from the Setka history."
    )

    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1.25, 1.25])
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
    )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted winner", pred["predicted_winner"])
    m2.metric(f"{player_a} win chance", format_percent(pred["player_a_win_probability"]))
    m3.metric(
        f"1st set Over {first_set_line}",
        format_percent(pred["first_set_over_probability"]),
    )
    m4.metric("Confidence", pred["confidence"], help="Based on player sample size, Elo availability, recent data, and H2H sample.")

    r1, r2 = st.columns([1.05, 1])
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

    line_cols = st.columns(4)
    line_cols[0].metric("Expected 1st-set points", format_number(pred["expected_first_set_points"], 2))
    line_cols[1].metric("Expected total points", format_number(pred["expected_total_points"], 2))
    line_cols[2].metric(
        f"Total Over {total_points_line}",
        format_percent(pred["total_points_over_probability"]),
    )
    line_cols[3].metric(
        f"Total Under {total_points_line}",
        format_percent(pred["total_points_under_probability"]),
    )

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
