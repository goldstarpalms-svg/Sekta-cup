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


# NOTE: Part 2B will add: Live, Analyze, Vault, and Edge pages below this line
# For now, only HOME page works. Other pages will show blank until Part 2B.
