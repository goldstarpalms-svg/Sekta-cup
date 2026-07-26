from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from src.ml_pipeline import load_model_bundle, predict_with_bundle
    from src.setka_core import load_raw_data
    from src.setka_live import (
        fetch_nearest_matches,
        fetch_live_matches,
        add_lagos_time,
        add_location_names,
        location_map,
    )
    from src.bankroll import (
        load_bankroll_state,
        save_bankroll_state,
        kelly_stake,
        fair_odds_from_probability,
        add_bet,
        settle_bet,
        calculate_daily_pnl,
        calculate_stats,
        format_currency,
        check_loss_limit,
    )
    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as exc:
    IMPORTS_OK = False
    IMPORT_ERROR = str(exc)

st.set_page_config(
    page_title="OracleBet 🔮",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0A0E27 0%, #1a1f4a 50%, #0A0E27 100%); color: #E8ECFF; }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    h1 { color: #00F5FF !important; text-shadow: 0 0 20px rgba(0, 245, 255, 0.5); font-weight: 900; }
    h2, h3, h4 { color: #E8ECFF !important; font-weight: 700; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0A0E27 0%, #151a3d 100%); border-right: 1px solid rgba(0, 245, 255, 0.2); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #00F5FF !important; }
    [data-testid="stMetric"] { background: rgba(0, 245, 255, 0.05); border: 1px solid rgba(0, 245, 255, 0.2); border-radius: 16px; padding: 1rem; box-shadow: 0 0 20px rgba(0, 245, 255, 0.1); }
    [data-testid="stMetricValue"] { color: #00F5FF !important; font-size: 1.8rem !important; font-weight: 900; }
    [data-testid="stMetricLabel"] { color: #B537FF !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem !important; }
    .stButton button { background: linear-gradient(135deg, #00F5FF 0%, #B537FF 100%); color: #0A0E27; border: none; border-radius: 12px; font-weight: 900; letter-spacing: 1px; padding: 0.6rem 1.5rem; box-shadow: 0 0 15px rgba(0, 245, 255, 0.4); }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 0 25px rgba(0, 245, 255, 0.6); }
    .stAlert { background: rgba(0, 245, 255, 0.08); border: 1px solid rgba(0, 245, 255, 0.3); border-radius: 12px; }
    .pick-card { background: linear-gradient(135deg, rgba(0, 245, 255, 0.05) 0%, rgba(181, 55, 255, 0.05) 100%); border: 1px solid rgba(0, 245, 255, 0.3); border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 0 30px rgba(0, 245, 255, 0.1); }
    .pick-card-top { background: linear-gradient(135deg, rgba(0, 255, 156, 0.1) 0%, rgba(0, 245, 255, 0.1) 100%); border: 2px solid #00FF9C; box-shadow: 0 0 40px rgba(0, 255, 156, 0.3); }
    .pick-title { font-size: 1.3rem; font-weight: 900; color: #00F5FF; margin-bottom: 0.5rem; }
    .pick-time { color: #B537FF; font-size: 0.9rem; font-weight: 600; margin-bottom: 1rem; }
    .pick-prediction { font-size: 1.8rem; font-weight: 900; color: #00FF9C; margin: 1rem 0; text-shadow: 0 0 15px rgba(0, 255, 156, 0.4); }
    .confidence-bar { background: rgba(255, 255, 255, 0.1); border-radius: 10px; height: 20px; overflow: hidden; margin: 0.5rem 0; }
    .confidence-fill { background: linear-gradient(90deg, #00F5FF 0%, #00FF9C 100%); height: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 255, 156, 0.5); }
    .pick-stat { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #E8ECFF; }
    .stat-label { color: #B537FF; font-weight: 600; }
    .stat-value { color: #00F5FF; font-weight: 700; }
    .ai-analysis { background: rgba(181, 55, 255, 0.1); border-left: 3px solid #B537FF; padding: 0.8rem 1rem; margin: 1rem 0; border-radius: 8px; font-style: italic; color: #E8ECFF; }
    .forecast-card { background: linear-gradient(135deg, rgba(255, 184, 0, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%); border: 1px solid rgba(255, 184, 0, 0.3); border-radius: 20px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem; }
    .forecast-value { font-size: 2.5rem; font-weight: 900; color: #FFB800; text-shadow: 0 0 20px rgba(255, 184, 0, 0.4); }
    .warning-card { background: rgba(255, 51, 102, 0.15); border: 2px solid #FF3366; border-radius: 16px; padding: 1rem; margin: 1rem 0; }
    [data-testid="collapsedControl"] { display: block !important; visibility: visible !important; background: linear-gradient(135deg, #00F5FF 0%, #B537FF 100%) !important; border-radius: 50% !important; padding: 8px !important; box-shadow: 0 0 20px rgba(0, 245, 255, 0.6) !important; position: fixed !important; top: 12px !important; left: 12px !important; z-index: 999999 !important; width: 44px !important; height: 44px !important; }
    [data-testid="collapsedControl"] svg { color: #0A0E27 !important; fill: #0A0E27 !important; }
    @media (max-width: 768px) { [data-testid="collapsedControl"] { width: 48px !important; height: 48px !important; } .main .block-container { padding-top: 4rem !important; } }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 🔮 ORACLEBET")
    st.caption("The Future of Predictions")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📡 Live", "🧠 Analyze", "🛡️ Vault", "📊 Edge"],
        label_visibility="collapsed",
    )
    st.divider()
    if IMPORTS_OK:
        state = load_bankroll_state()
        with st.expander("⚙️ Settings", expanded=False):
            new_bankroll = st.number_input("Bankroll (₦)", 1000, 10000000, int(state["current_bankroll"]), 1000)
            new_daily_limit = st.slider("Daily Loss Limit (%)", 5, 25, int(state["daily_loss_limit_pct"]))
            new_kelly = st.slider("Kelly Fraction", 0.10, 1.00, float(state["kelly_fraction"]), 0.05)
            new_min_conf = st.slider("Min Confidence (%)", 55, 80, int(state["min_confidence"] * 100))
            if st.button("💾 Save Settings"):
                state["current_bankroll"] = new_bankroll
                state["daily_loss_limit_pct"] = new_daily_limit
                state["kelly_fraction"] = new_kelly
                state["min_confidence"] = new_min_conf / 100
                save_bankroll_state(state)
                st.success("✅ Saved!")
                st.rerun()
    st.divider()
    st.caption("💎 155K+ matches")
    st.caption("🤖 XGBoost ML")


if not IMPORTS_OK:
    st.error(f"⚠️ Import Error: {IMPORT_ERROR}")
    st.stop()


MODEL_PATH = Path("models/setka_ml_bundle.joblib")


@st.cache_resource(show_spinner="🔮 Loading Oracle...")
def get_model_bundle():
    if MODEL_PATH.exists():
        try:
            return load_model_bundle(MODEL_PATH), None
        except Exception as exc:
            return None, str(exc)
    return None, "No trained model found."


@st.cache_data(ttl=60, show_spinner="⚡ Fetching matches...")
def get_upcoming_matches():
    try:
        locs = location_map()
        frame = fetch_nearest_matches()
        frame = add_lagos_time(frame)
        frame = add_location_names(frame, locs)
        return frame, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=15, show_spinner="📡 Live matches...")
def get_live_matches():
    try:
        locs = location_map()
        frame = fetch_live_matches()
        frame = add_lagos_time(frame)
        frame = add_location_names(frame, locs)
        return frame, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def generate_ai_reasoning(pred, player_a, player_b):
    features = pred.get("features", {})
    reasons = []
    win_streak_a = features.get("a_win_streak", 0)
    win_streak_b = features.get("b_win_streak", 0)
    if win_streak_a >= 3:
        reasons.append(f"🔥 {player_a} on {int(win_streak_a)}-win streak")
    if win_streak_b >= 3:
        reasons.append(f"🔥 {player_b} on {int(win_streak_b)}-win streak")
    hours_a = features.get("a_hours_since_last", 168)
    hours_b = features.get("b_hours_since_last", 168)
    if hours_a < 2:
        reasons.append(f"😴 {player_a} may be tired ({hours_a:.1f}h ago)")
    if hours_b < 2:
        reasons.append(f"😴 {player_b} may be tired ({hours_b:.1f}h ago)")
    elo_diff = features.get("elo_diff", 0)
    if abs(elo_diff) > 100:
        stronger = player_a if elo_diff > 0 else player_b
        reasons.append(f"💪 {stronger} +{abs(elo_diff):.0f} Elo advantage")
    h2h = features.get("h2h_matches", 0)
    h2h_rate = features.get("h2h_a_win_rate", 0.5)
    if h2h >= 3:
        if h2h_rate > 0.65:
            reasons.append(f"⚔️ {player_a} owns H2H ({h2h_rate*100:.0f}% of {int(h2h)})")
        elif h2h_rate < 0.35:
            reasons.append(f"⚔️ {player_b} owns H2H ({(1-h2h_rate)*100:.0f}% of {int(h2h)})")
    matches_today_a = features.get("a_matches_today", 0)
    matches_today_b = features.get("b_matches_today", 0)
    if matches_today_a >= 4:
        reasons.append(f"⚠️ {player_a} played {int(matches_today_a)} today")
    if matches_today_b >= 4:
        reasons.append(f"⚠️ {player_b} played {int(matches_today_b)} today")
    if not reasons:
        reasons.append("📊 Based on 155K+ matches analysis")
    return " • ".join(reasons[:3])


# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    state = load_bankroll_state()
    current_bankroll = state["current_bankroll"]
    starting_bankroll = state["starting_bankroll"]
    
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    
    st.markdown("# 🔮 ORACLEBET")
    st.markdown(f"### {greeting}, Champion 👑")
    
    limit_check = check_loss_limit(state)
    if limit_check["exceeded"]:
        st.markdown(f"""
        <div class="warning-card">
            🛑 <strong>DAILY LOSS LIMIT REACHED</strong><br>
            Lost {format_currency(abs(limit_check['loss']))} today (limit: {format_currency(abs(limit_check['limit']))})<br>
            <em>Take a break. Come back tomorrow.</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    stats = calculate_stats(state)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_pnl = calculate_daily_pnl(state, today_str)
    total_pnl = current_bankroll - starting_bankroll
    total_pnl_pct = (total_pnl / starting_bankroll * 100) if starting_bankroll > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bankroll", format_currency(current_bankroll), format_currency(total_pnl) if total_pnl != 0 else None)
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.0f}%", f"{stats['settled_bets']} bets")
    with col3:
        st.metric("Total ROI", f"{total_pnl_pct:+.1f}%", format_currency(today_pnl) if today_pnl != 0 else "Today: ₦0")
    with col4:
        st.metric("Pending", str(stats['pending_bets']), "🔥" if stats['pending_bets'] > 0 else "None")
    
    st.markdown("---")
    
    bundle, model_error = get_model_bundle()
    matches, match_error = get_upcoming_matches()
    
    if model_error:
        st.warning(f"⚠️ {model_error}")
        st.stop()
    
    if match_error:
        st.warning(f"⚠️ {match_error}")
    
    predictions = []
    if not matches.empty and bundle:
        for _, match in matches.head(30).iterrows():
            try:
                pred = predict_with_bundle(bundle, match["player1"], match["player2"], current_dt=pd.Timestamp.now())
                confidence = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
                if confidence >= state["min_confidence"]:
                    predictions.append({
                        "match": f"{match['player1']} vs {match['player2']}",
                        "player1": match["player1"],
                        "player2": match["player2"],
                        "time": match.get("start_time_lagos", "TBD"),
                        "location": match.get("location", "Setka Cup"),
                        "winner": pred["predicted_winner"],
                        "confidence": confidence,
                        "prediction": pred,
                    })
            except Exception:
                continue
    
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_picks = predictions[:5]
    
    forecast_label = "HIGH ✨" if len(top_picks) >= 3 else "MEDIUM" if len(top_picks) >= 1 else "LOW 🌧️"
    st.markdown(f"""
    <div class="forecast-card">
        <div style="color: #FFB800; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">☀️ Oracle Forecast</div>
        <div class="forecast-value">{len(top_picks)} STRONG PICKS</div>
        <div style="color: #E8ECFF; margin-top: 0.5rem;">Confidence: {forecast_label}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not top_picks:
        st.info("🌙 No high-confidence picks now. Check back later!")
        st.stop()
    
    st.markdown("### ⭐ TOP PICK")
    top = top_picks[0]
    fair_odds = fair_odds_from_probability(top["confidence"])
    assumed_market_odds = fair_odds * 0.95
    kelly_result = kelly_stake(current_bankroll, top["confidence"], assumed_market_odds, state["kelly_fraction"], state["min_stake"], state["max_stake_pct"])
    suggested_stake = kelly_result["stake"]
    potential_win = suggested_stake * assumed_market_odds if suggested_stake > 0 else 0
    ai_reason = generate_ai_reasoning(top["prediction"], top["player1"], top["player2"])
    confidence_pct = top["confidence"] * 100
    
    st.markdown(f"""
    <div class="pick-card pick-card-top">
        <div class="pick-title">🏓 {top["match"]}</div>
        <div class="pick-time">⏰ {top["time"]} • {top["location"]}</div>
        <div class="pick-prediction">🎯 {top["winner"]} WINS</div>
        <div class="confidence-bar"><div class="confidence-fill" style="width: {confidence_pct}%"></div></div>
        <div style="text-align: center; color: #00FF9C; font-weight: 700; margin-bottom: 1rem;">{confidence_pct:.1f}% confidence</div>
        <div class="pick-stat"><span class="stat-label">💎 Fair Odds</span><span class="stat-value">{fair_odds:.2f}</span></div>
        <div class="pick-stat"><span class="stat-label">🎰 Est. SportyBet</span><span class="stat-value">{assumed_market_odds:.2f}</span></div>
        <div class="pick-stat"><span class="stat-label">💰 Kelly Stake</span><span class="stat-value">{format_currency(suggested_stake)}</span></div>
        <div class="pick-stat"><span class="stat-label">📊 Potential Win</span><span class="stat-value">{format_currency(potential_win)}</span></div>
        <div class="pick-stat"><span class="stat-label">🎯 Edge</span><span class="stat-value">{kelly_result['kelly_pct']}%</span></div>
        <div class="ai-analysis">🧠 <strong>AI:</strong> {ai_reason}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if suggested_stake > 0:
        if st.button(f"📝 Track This Bet ({format_currency(suggested_stake)})", key="track_top"):
            add_bet(state, {
                "match": top["match"],
                "player1": top["player1"],
                "player2": top["player2"],
                "prediction": top["winner"],
                "confidence": top["confidence"],
                "stake": suggested_stake,
                "odds": assumed_market_odds,
                "market": "match_winner",
            })
            st.success("✅ Tracked in Vault!")
            st.rerun()
    
    if len(top_picks) > 1:
        st.markdown(f"### 📋 More Picks ({len(top_picks) - 1})")
        for i, pick in enumerate(top_picks[1:]):
            conf_pct = pick["confidence"] * 100
            odds = fair_odds_from_probability(pick["confidence"])
            market_odds = odds * 0.95
            kelly = kelly_stake(current_bankroll, pick["confidence"], market_odds, state["kelly_fraction"], state["min_stake"], state["max_stake_pct"])
            reason = generate_ai_reasoning(pick["prediction"], pick["player1"], pick["player2"])
            
            st.markdown(f"""
            <div class="pick-card">
                <div class="pick-title">🏓 {pick["match"]}</div>
                <div class="pick-time">⏰ {pick["time"]} • {pick["location"]}</div>
                <div class="pick-prediction" style="font-size: 1.4rem;">🎯 {pick["winner"]} ({conf_pct:.1f}%)</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width: {conf_pct}%"></div></div>
                <div class="pick-stat"><span class="stat-label">Odds / Stake</span><span class="stat-value">{market_odds:.2f} / {format_currency(kelly['stake'])}</span></div>
                <div class="ai-analysis" style="font-size: 0.85rem;">🧠 {reason}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if kelly['stake'] > 0:
                if st.button(f"📝 Track ({format_currency(kelly['stake'])})", key=f"track_more_{i}"):
                    add_bet(state, {
                        "match": pick["match"],
                        "player1": pick["player1"],
                        "player2": pick["player2"],
                        "prediction": pick["winner"],
                        "confidence": pick["confidence"],
                        "stake": kelly['stake'],
                        "odds": market_odds,
                        "market": "match_winner",
                    })
                    st.success("✅ Tracked!")
                    st.rerun()



# ============================================
# LIVE PAGE
# ============================================
elif page == "📡 Live":
    st.markdown("# 📡 LIVE MATCHES")
    st.caption("Real-time Setka Cup matches")
    
    live_matches, live_error = get_live_matches()
    
    if live_error:
        st.warning(f"⚠️ {live_error}")
    
    if live_matches.empty:
        st.info("😴 No live matches now. Check during match hours.")
    else:
        st.success(f"🔴 {len(live_matches)} matches LIVE")
        bundle, _ = get_model_bundle()
        
        for _, match in live_matches.iterrows():
            p1 = match.get("player1", "?")
            p2 = match.get("player2", "?")
            score = match.get("score", "")
            
            st.markdown(f"""
            <div class="pick-card">
                <div class="pick-title">🔴 LIVE: {p1} vs {p2}</div>
                <div class="pick-time">Score: {score}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if bundle:
                try:
                    pred = predict_with_bundle(bundle, p1, p2)
                    conf = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
                    st.caption(f"🎯 Model: {pred['predicted_winner']} wins ({conf*100:.1f}%)")
                except Exception:
                    pass


# ============================================
# ANALYZE PAGE
# ============================================
elif page == "🧠 Analyze":
    st.markdown("# 🧠 DEEP ANALYZE")
    st.caption("Analyze any matchup")
    
    bundle, model_error = get_model_bundle()
    
    if model_error:
        st.warning(f"⚠️ {model_error}")
        st.stop()
    
    try:
        raw_matches, _ = load_raw_data()
        players = sorted(set(raw_matches["player1"].dropna().unique()) | set(raw_matches["player2"].dropna().unique()))
    except Exception:
        players = []
    
    if not players:
        st.error("Could not load players")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Player A", players, key="pa")
    with col2:
        player_b = st.selectbox("Player B", players, index=1 if len(players) > 1 else 0, key="pb")
    
    if player_a == player_b:
        st.warning("Select two different players")
        st.stop()
    
    if st.button("🔮 ANALYZE"):
        try:
            pred = predict_with_bundle(bundle, player_a, player_b)
            conf = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
            reason = generate_ai_reasoning(pred, player_a, player_b)
            fair = fair_odds_from_probability(conf)
            
            st.markdown(f"""
            <div class="pick-card pick-card-top">
                <div class="pick-title">🏓 {player_a} vs {player_b}</div>
                <div class="pick-prediction">🎯 {pred['predicted_winner']} WINS</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width: {conf*100}%"></div></div>
                <div style="text-align: center; color: #00FF9C; font-weight: 700; margin-bottom: 1rem;">{conf*100:.1f}% confidence</div>
                <div class="pick-stat"><span class="stat-label">Fair Odds</span><span class="stat-value">{fair:.2f}</span></div>
                <div class="ai-analysis">🧠 <strong>Analysis:</strong> {reason}</div>
            </div>
            """, unsafe_allow_html=True)
            
            features = pred.get("features", {})
            st.markdown("### 📊 Detailed Metrics")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"{player_a} Elo", f"{features.get('a_elo', 0):.0f}")
                st.metric(f"{player_a} Form", f"{features.get('a_weighted_recent_form', 0)*100:.0f}%")
                st.metric(f"{player_a} Streak", f"{int(features.get('a_win_streak', 0))}")
            with c2:
                st.metric(f"{player_b} Elo", f"{features.get('b_elo', 0):.0f}")
                st.metric(f"{player_b} Form", f"{features.get('b_weighted_recent_form', 0)*100:.0f}%")
                st.metric(f"{player_b} Streak", f"{int(features.get('b_win_streak', 0))}")
            
            st.markdown("### 🔍 H2H History")
            h2h_matches = features.get("h2h_matches", 0)
            h2h_a_rate = features.get("h2h_a_win_rate", 0.5)
            
            if h2h_matches > 0:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("H2H Matches", f"{int(h2h_matches)}")
                with c2:
                    st.metric(f"{player_a} Wins", f"{h2h_a_rate*100:.0f}%")
                with c3:
                    st.metric(f"{player_b} Wins", f"{(1-h2h_a_rate)*100:.0f}%")
            else:
                st.info("No head-to-head history between these players")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


# ============================================
# VAULT PAGE
# ============================================
elif page == "🛡️ Vault":
    st.markdown("# 🛡️ BANKROLL VAULT")
    st.caption("Track your paper trading performance")
    
    state = load_bankroll_state()
    stats = calculate_stats(state)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Bankroll", format_currency(state["current_bankroll"]))
        st.metric("Starting", format_currency(state["starting_bankroll"]))
    with c2:
        st.metric("Total Profit", format_currency(stats["total_profit"]))
        st.metric("Total Staked", format_currency(stats["total_staked"]))
    with c3:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        st.metric("ROI", f"{stats['roi']:+.1f}%")
    
    st.markdown("---")
    
    # Streak info
    streak = state.get("streak", {"current": 0, "type": "none"})
    if streak.get("type") != "none":
        emoji = "🔥" if streak["type"] == "win" else "❄️"
        st.markdown(f"### {emoji} Current Streak: {streak['current']} {streak['type']}s")
    
    st.markdown("### 📋 Bet Journal")
    
    if not state["bets"]:
        st.info("💡 No bets tracked yet. Go to Home page and tap 'Track This Bet' on picks.")
    else:
        pending = [b for b in state["bets"] if b["status"] == "pending"]
        if pending:
            st.markdown(f"#### ⏳ Pending Bets ({len(pending)})")
            for bet in reversed(pending):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{bet['match']}** → {bet['prediction']}")
                    st.caption(f"Stake: {format_currency(bet['stake'])} @ {bet['odds']:.2f} | Confidence: {bet['confidence']*100:.1f}%")
                with col2:
                    if st.button("✅ Won", key=f"win_{bet['id']}"):
                        settle_bet(state, bet["id"], True)
                        st.success("Bet settled as WIN!")
                        st.rerun()
                with col3:
                    if st.button("❌ Lost", key=f"lose_{bet['id']}"):
                        settle_bet(state, bet["id"], False)
                        st.error("Bet settled as LOSS")
                        st.rerun()
        
        settled = [b for b in state["bets"] if b["status"] == "settled"]
        if settled:
            st.markdown(f"#### 📜 History ({len(settled)} settled)")
            
            # Recent history table
            for bet in reversed(settled[-20:]):
                emoji = "✅" if bet.get("won") else "❌"
                pnl = bet.get("profit_loss", 0)
                color = "#00FF9C" if pnl > 0 else "#FF3366"
                date = bet["timestamp"][:10]
                st.markdown(f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {emoji} <strong>{bet['match']}</strong> → {bet['prediction']}<br>
                    <span style="color: #B537FF; font-size: 0.85rem;">{date} • Stake: {format_currency(bet.get('stake', 0))} @ {bet.get('odds', 0):.2f}</span> 
                    <span style="color:{color}; float: right; font-weight: 900;">{format_currency(pnl)}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Reset option
        st.markdown("---")
        with st.expander("⚠️ Danger Zone"):
            st.warning("This will reset ALL bets and bankroll to starting values.")
            if st.button("🗑️ Reset All Data"):
                if state.get("bets"):
                    state["bets"] = []
                    state["current_bankroll"] = state["starting_bankroll"]
                    state["streak"] = {"current": 0, "type": "none", "best_win": 0, "worst_loss": 0}
                    save_bankroll_state(state)
                    st.success("✅ Reset complete!")
                    st.rerun()


# ============================================
# EDGE PAGE
# ============================================
elif page == "📊 Edge":
    st.markdown("# 📊 STATISTICAL EDGE")
    st.caption("Model transparency and performance")
    
    bundle, model_error = get_model_bundle()
    
    if model_error:
        st.warning(f"⚠️ {model_error}")
        st.stop()
    
    st.markdown("### 🤖 Model Info")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Algorithm", bundle.get("algorithm", "unknown").upper())
    with c2:
        st.metric("Train Rows", f"{bundle.get('train_rows', 0):,}")
    with c3:
        st.metric("Test Rows", f"{bundle.get('test_rows', 0):,}")
    
    st.markdown("### 📈 Performance Metrics")
    metrics = bundle.get("metrics", [])
    if metrics:
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True)
        
        # Highlight winner accuracy
        for m in metrics:
            if m.get("model") == "winner" and m.get("accuracy"):
                acc_pct = m["accuracy"] * 100
                roc = m.get("roc_auc", 0)
                
                if acc_pct >= 65:
                    st.success(f"🎯 Winner Accuracy: {acc_pct:.1f}% (ROC-AUC: {roc:.3f}) - EXCELLENT")
                elif acc_pct >= 60:
                    st.info(f"🎯 Winner Accuracy: {acc_pct:.1f}% (ROC-AUC: {roc:.3f}) - GOOD")
                else:
                    st.warning(f"🎯 Winner Accuracy: {acc_pct:.1f}% (ROC-AUC: {roc:.3f}) - NEEDS IMPROVEMENT")
    else:
        st.info("No metrics available. Retrain the model to see metrics.")
    
    st.markdown("### 🎯 Features Being Used")
    features_list = bundle.get("feature_columns", [])
    if features_list:
        st.write(f"**Total features:** {len(features_list)}")
        
        # Categorize features
        categories = {
            "Elo Ratings": [f for f in features_list if "elo" in f.lower()],
            "Win Rates": [f for f in features_list if "win_rate" in f.lower() or "recent_form" in f.lower()],
            "Fatigue": [f for f in features_list if "hours" in f.lower() or "matches_today" in f.lower()],
            "Streaks": [f for f in features_list if "streak" in f.lower()],
            "H2H": [f for f in features_list if "h2h" in f.lower()],
            "Points/Sets": [f for f in features_list if "points" in f.lower() or "set" in f.lower()],
        }
        
        for cat, feats in categories.items():
            if feats:
                with st.expander(f"{cat} ({len(feats)})"):
                    for f in feats:
                        st.text(f"• {f}")
    
    st.markdown("### 💡 About the Oracle")
    st.markdown("""
    <div class="pick-card">
        <strong>🔮 How OracleBet Works:</strong><br><br>
        • Analyzes <strong>155,000+ historical matches</strong><br>
        • Uses <strong>XGBoost machine learning</strong> algorithm<br>
        • Tracks <strong>60+ features</strong> per match<br>
        • Updates <strong>Elo ratings dynamically</strong><br>
        • Detects <strong>fatigue, streaks, and momentum</strong><br>
        • Considers <strong>head-to-head history</strong><br>
        • Calculates <strong>Kelly Criterion</strong> for optimal stakes<br>
        • Only shows picks with your <strong>minimum confidence</strong> threshold<br><br>
        <em>Remember: No model is perfect. Always paper trade first, bet responsibly, and never chase losses.</em>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🔮 OracleBet v1.0 • Built with 155K+ matches • Paper trade first, bet second")
