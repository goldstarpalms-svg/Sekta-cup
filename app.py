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
    from src.predictions_tracker import (
        load_predictions,
        add_prediction,
        update_prediction_result,
        calculate_track_record,
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
    /* ===== ENHANCED HAMBURGER MENU BUTTON ===== */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #00F5FF 0%, #B537FF 100%) !important;
        border-radius: 14px !important;
        padding: 12px !important;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.7), 0 0 60px rgba(181, 55, 255, 0.4) !important;
        position: fixed !important;
        top: 18px !important;
        left: 18px !important;
        z-index: 9999999 !important;
        width: 56px !important;
        height: 56px !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        color: #0A0E27 !important;
        fill: #0A0E27 !important;
        width: 30px !important;
        height: 30px !important;
        stroke: #0A0E27 !important;
        stroke-width: 3 !important;
    }
    
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 0 40px rgba(0, 245, 255, 1), 0 0 80px rgba(181, 55, 255, 0.7) !important;
    }
    
    @keyframes hamburger-pulse {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.7), 0 0 60px rgba(181, 55, 255, 0.4);
        }
        50% { 
            box-shadow: 0 0 40px rgba(0, 245, 255, 1), 0 0 80px rgba(181, 55, 255, 0.7);
        }
    }
    
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        animation: hamburger-pulse 2.5s infinite !important;
    }
    
    /* Push main content down to avoid overlap */
    .main .block-container {
        padding-top: 5rem !important;
    }
    
    /* Roadmap-specific styles */
    .roadmap-phase {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.05) 0%, rgba(181, 55, 255, 0.05) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .roadmap-done {
        border-left: 4px solid #00FF9C;
    }
    
    .roadmap-progress {
        border-left: 4px solid #FFB800;
    }
    
    .roadmap-future {
        border-left: 4px solid #B537FF;
        opacity: 0.7;
    }
    
    .roadmap-title {
        font-size: 1.1rem;
        font-weight: 900;
        color: #00F5FF;
        margin-bottom: 0.3rem;
    }
    
    .roadmap-status {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .status-done {
        background: rgba(0, 255, 156, 0.2);
        color: #00FF9C;
    }
    
    .status-progress {
        background: rgba(255, 184, 0, 0.2);
        color: #FFB800;
    }
    
    .status-future {
        background: rgba(181, 55, 255, 0.2);
        color: #B537FF;
    }
    
    .roadmap-features {
        margin-top: 0.5rem;
        color: #E8ECFF;
    }
    
    .roadmap-features li {
        margin: 0.3rem 0;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* ===== CUSTOM BOTTOM NAVIGATION BAR (MOBILE-FIRST) ===== */
    .bottom-nav {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: rgba(10, 14, 39, 0.98) !important;
        border-top: 2px solid rgba(0, 245, 255, 0.4) !important;
        padding: 8px 4px !important;
        z-index: 999999 !important;
        display: flex !important;
        justify-content: space-around !important;
        align-items: center !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 -10px 40px rgba(0, 245, 255, 0.3) !important;
    }
    
    .bottom-nav-item {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 6px 4px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-width: 44px !important;
        text-decoration: none !important;
        border-radius: 12px !important;
    }
    
    .bottom-nav-item:hover {
        background: rgba(0, 245, 255, 0.1) !important;
    }
    
    .bottom-nav-item.active {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(181, 55, 255, 0.2) 100%) !important;
    }
    
    .bottom-nav-icon {
        font-size: 20px !important;
        margin-bottom: 2px !important;
        display: block !important;
    }
    
    .bottom-nav-label {
        font-size: 9px !important;
        color: #E8ECFF !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        white-space: nowrap !important;
    }
    
    .bottom-nav-item.active .bottom-nav-label {
        color: #00F5FF !important;
    }
    
    /* Add padding to bottom of page so content isn't hidden behind nav */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Hide default streamlit sidebar toggle on mobile since we have our own nav */
    @media (max-width: 768px) {
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
    }
    
    /* Desktop: Keep default sidebar toggle */
    @media (min-width: 769px) {
        .bottom-nav {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 🔮 ORACLEBET")
    st.caption("The Future of Predictions")
    st.divider()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    nav_options = ["🏠 Home", "🎯 All Picks", "📡 Live", "🧠 Analyze", "📈 Track Record", "🛡️ Vault", "📊 Edge", "🗺️ Roadmap"]
    
    try:
        current_index = nav_options.index(st.session_state.current_page)
    except ValueError:
        current_index = 0
    
    sidebar_page = st.radio(
        "Navigation",
        nav_options,
        index=current_index,
        label_visibility="collapsed",
        key="sidebar_nav_selector",
    )
    
    # Sync sidebar selection to session state
    if sidebar_page != st.session_state.current_page:
        st.session_state.current_page = sidebar_page
        st.rerun()
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


# ===== CUSTOM BOTTOM NAVIGATION (MOBILE) =====

# Initialize session state for page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# Get current page from URL query params if available
query_params = st.query_params
if "page" in query_params:
    page_from_url = query_params["page"]
    valid_pages = ["Home", "AllPicks", "Live", "Analyze", "TrackRecord", "Vault", "Edge", "Roadmap"]
    page_map = {
        "Home": "🏠 Home",
        "AllPicks": "🎯 All Picks",
        "Live": "📡 Live",
        "Analyze": "🧠 Analyze",
        "TrackRecord": "📈 Track Record",
        "Vault": "🛡️ Vault",
        "Edge": "📊 Edge",
        "Roadmap": "🗺️ Roadmap",
    }
    if page_from_url in page_map:
        st.session_state.current_page = page_map[page_from_url]

# Determine active page for highlighting
current = st.session_state.current_page

def is_active(page_name):
    return "active" if current == page_name else ""

# Render bottom navigation bar
st.markdown(f"""
<div class="bottom-nav">
    <a href="?page=Home" class="bottom-nav-item {is_active('🏠 Home')}" target="_self">
        <div class="bottom-nav-icon">🏠</div>
        <div class="bottom-nav-label">Home</div>
    </a>
    <a href="?page=AllPicks" class="bottom-nav-item {is_active('🎯 All Picks')}" target="_self">
        <div class="bottom-nav-icon">🎯</div>
        <div class="bottom-nav-label">Picks</div>
    </a>
    <a href="?page=Live" class="bottom-nav-item {is_active('📡 Live')}" target="_self">
        <div class="bottom-nav-icon">📡</div>
        <div class="bottom-nav-label">Live</div>
    </a>
    <a href="?page=Analyze" class="bottom-nav-item {is_active('🧠 Analyze')}" target="_self">
        <div class="bottom-nav-icon">🧠</div>
        <div class="bottom-nav-label">Analyze</div>
    </a>
    <a href="?page=TrackRecord" class="bottom-nav-item {is_active('📈 Track Record')}" target="_self">
        <div class="bottom-nav-icon">📈</div>
        <div class="bottom-nav-label">Record</div>
    </a>
    <a href="?page=Vault" class="bottom-nav-item {is_active('🛡️ Vault')}" target="_self">
        <div class="bottom-nav-icon">🛡️</div>
        <div class="bottom-nav-label">Vault</div>
    </a>
    <a href="?page=Edge" class="bottom-nav-item {is_active('📊 Edge')}" target="_self">
        <div class="bottom-nav-icon">📊</div>
        <div class="bottom-nav-label">Edge</div>
    </a>
    <a href="?page=Roadmap" class="bottom-nav-item {is_active('🗺️ Roadmap')}" target="_self">
        <div class="bottom-nav-icon">🗺️</div>
        <div class="bottom-nav-label">Map</div>
    </a>
</div>
""", unsafe_allow_html=True)

# Use session state page as the active page
page = st.session_state.current_page


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
# LIVE PAGE (UPGRADED)
# ============================================
elif page == "📡 Live":
    st.markdown("# 📡 LIVE MATCHES")
    st.caption("Real-time Setka Cup matches with live scores")
    
    # Auto-refresh option
    refresh_col1, refresh_col2 = st.columns([3, 1])
    with refresh_col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh (15s)", value=False)
    
    if auto_refresh:
        st.markdown("""
        <meta http-equiv="refresh" content="15">
        """, unsafe_allow_html=True)
    
    live_matches, live_error = get_live_matches()
    
    if live_error:
        st.warning(f"⚠️ {live_error}")
    
    if live_matches.empty:
        st.info("😴 No live matches right now. Check during peak Setka Cup hours (usually 24/7 but varies).")
        st.markdown("""
        <div class="pick-card">
            <div class="pick-title">💡 About Live Matches</div>
            <div style="color: #E8ECFF;">
                Live matches appear here when players are actively competing.<br>
                We show:<br>
                • Current score (sets and points)<br>
                • Live probability updates<br>
                • AI recommendations<br>
                • Set-by-set breakdown
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"🔴 {len(live_matches)} matches LIVE right now")
        
        bundle, _ = get_model_bundle()
        
        for idx, match in live_matches.iterrows():
            p1 = match.get("player1", "?")
            p2 = match.get("player2", "?")
            score = match.get("score", "0:0")
            set_scores = match.get("set_scores", "")
            location = match.get("location", "Setka Cup")
            
            # Parse current score
            try:
                p1_sets, p2_sets = score.split(":") if ":" in str(score) else ("0", "0")
                p1_sets = int(p1_sets.strip())
                p2_sets = int(p2_sets.strip())
            except:
                p1_sets, p2_sets = 0, 0
            
            # Determine leader
            if p1_sets > p2_sets:
                leader = p1
                lead_color = "#00FF9C"
            elif p2_sets > p1_sets:
                leader = p2
                lead_color = "#00FF9C"
            else:
                leader = "Tied"
                lead_color = "#FFB800"
            
            st.markdown(f"""
            <div class="pick-card" style="border-left: 4px solid #FF3366;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="pick-title">🔴 LIVE</div>
                    <div style="color: #FF3366; font-weight: 900; font-size: 0.85rem;">● BROADCASTING</div>
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #E8ECFF; margin-bottom: 0.5rem;">
                    {p1} vs {p2}
                </div>
                <div class="pick-time">📍 {location}</div>
                
                <div style="background: rgba(0, 245, 255, 0.1); padding: 1rem; border-radius: 12px; margin: 1rem 0; text-align: center;">
                    <div style="color: #B537FF; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Current Score</div>
                    <div style="font-size: 2.5rem; font-weight: 900; color: #00F5FF; margin: 0.5rem 0;">
                        {p1_sets} : {p2_sets}
                    </div>
                    <div style="color: {lead_color}; font-weight: 700;">
                        {'🏆 ' + leader + ' leads' if leader != 'Tied' else '⚖️ Tied'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show set-by-set if available
            if set_scores:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                    <div style="color: #B537FF; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Set-by-Set</div>
                    <div style="color: #E8ECFF; font-family: monospace; font-size: 1rem; margin-top: 0.3rem;">{set_scores}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Live Prediction
            if bundle:
                try:
                    pred = predict_with_bundle(bundle, p1, p2)
                    conf = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
                    predicted_winner = pred["predicted_winner"]
                    
                    # Adjust based on current score (basic)
                    if p1_sets > p2_sets and predicted_winner == p1:
                        confidence_note = "✅ Prediction on track"
                    elif p2_sets > p1_sets and predicted_winner == p2:
                        confidence_note = "✅ Prediction on track"
                    elif p1_sets == p2_sets:
                        confidence_note = "⏳ Match too early to tell"
                    else:
                        confidence_note = "⚠️ Underdog leading - upset possible"
                    
                    st.markdown(f"""
                    <div style="background: rgba(181, 55, 255, 0.1); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                        <div style="color: #B537FF; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">🧠 AI Prediction</div>
                        <div style="color: #00FF9C; font-weight: 700; font-size: 1.1rem; margin: 0.3rem 0;">
                            {predicted_winner} to win ({conf*100:.1f}%)
                        </div>
                        <div style="color: #E8ECFF; font-size: 0.9rem;">{confidence_note}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    pass
            
            st.markdown("</div>", unsafe_allow_html=True)


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
# ALL PICKS PAGE
# ============================================
elif page == "🎯 All Picks":
    st.markdown("# 🎯 ALL UPCOMING PICKS")
    st.caption("Complete predictions across all markets")
    
    state = load_bankroll_state()
    current_bankroll = state["current_bankroll"]
    
    bundle, model_error = get_model_bundle()
    matches, match_error = get_upcoming_matches()
    
    if model_error:
        st.warning(f"⚠️ ML Model: {model_error}")
    
    if match_error:
        st.warning(f"⚠️ Matches: {match_error}")
        st.stop()
    
    if matches.empty:
        st.info("😴 No upcoming matches found. Check back during match hours.")
        st.stop()
    
    # ===== FILTERS =====
    st.markdown("### 🔍 Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_conf_filter = st.slider(
            "Min Winner Confidence (%)",
            min_value=50,
            max_value=90,
            value=int(state["min_confidence"] * 100),
            step=5,
        )
    
    with filter_col2:
        market_filter = st.multiselect(
            "Show Markets",
            ["Winner", "Total Points", "First Set O/U", "Sets O/U"],
            default=["Winner", "Total Points", "First Set O/U", "Sets O/U"],
        )
    
    with filter_col3:
        limit = st.selectbox(
            "Max Matches to Show",
            [10, 25, 50, 100, "All"],
            index=1,
        )
    
    st.markdown("---")
    
    # ===== IMPORT RULE-BASED PREDICTOR for extra markets =====
    try:
        from src.setka_core import predict_match, build_context
        rule_ctx = build_context(load_raw_data()[0], load_raw_data()[1])
        rule_available = True
    except Exception as exc:
        rule_available = False
        st.warning(f"Rule model unavailable: {exc}")
    
    # ===== GENERATE ALL PREDICTIONS =====
    all_predictions = []
    max_matches = len(matches) if limit == "All" else int(limit)
    
    with st.spinner(f"🔮 Analyzing {min(max_matches, len(matches))} matches..."):
        for _, match in matches.head(max_matches).iterrows():
            try:
                p1 = match["player1"]
                p2 = match["player2"]
                
                # ML prediction (Winner)
                ml_pred = None
                if bundle:
                    try:
                        ml_pred = predict_with_bundle(bundle, p1, p2, current_dt=pd.Timestamp.now())
                    except Exception:
                        pass
                
                # Rule-based prediction (for Total Points, First Set, Sets)
                rule_pred = None
                if rule_available:
                    try:
                        rule_pred = predict_match(
                            p1, p2,
                            rule_ctx["player_stats"],
                            rule_ctx["matches"],
                            rule_ctx["global_stats"],
                            first_set_line=18.5,
                            total_points_line=75.5,
                            sets_line=3.5,
                        )
                    except Exception:
                        pass
                
                # Combine predictions
                if not ml_pred and not rule_pred:
                    continue
                
                # Winner (use ML if available, else rule)
                if ml_pred:
                    winner_prob = max(ml_pred["player_a_win_probability"], ml_pred["player_b_win_probability"])
                    predicted_winner = ml_pred["predicted_winner"]
                else:
                    winner_prob = max(rule_pred["player_a_win_probability"], rule_pred["player_b_win_probability"])
                    predicted_winner = rule_pred["predicted_winner"]
                
                # Filter by min confidence
                if winner_prob * 100 < min_conf_filter:
                    continue
                
                pred_data = {
                    "match": f"{p1} vs {p2}",
                    "player1": p1,
                    "player2": p2,
                    "time": match.get("start_time_lagos", "TBD"),
                    "date": match.get("start_date_lagos", ""),
                    "location": match.get("location", "Setka Cup"),
                    "winner_pred": predicted_winner,
                    "winner_prob": winner_prob,
                    "ml_pred": ml_pred,
                    "rule_pred": rule_pred,
                }
                
                # Add totals if rule model available
                if rule_pred:
                    total_over = rule_pred.get("total_points_over_probability", 0.5)
                    total_under = rule_pred.get("total_points_under_probability", 0.5)
                    pred_data["total_pick"] = "Over 75.5" if total_over > total_under else "Under 75.5"
                    pred_data["total_prob"] = max(total_over, total_under)
                    pred_data["expected_total"] = rule_pred.get("expected_total_points", 0)
                    
                    first_over = rule_pred.get("first_set_over_probability", 0.5)
                    first_under = rule_pred.get("first_set_under_probability", 0.5)
                    pred_data["first_pick"] = "Over 18.5" if first_over > first_under else "Under 18.5"
                    pred_data["first_prob"] = max(first_over, first_under)
                    pred_data["expected_first"] = rule_pred.get("expected_first_set_points", 0)
                    
                    sets_over = rule_pred.get("sets_over_probability", 0.5)
                    sets_under = rule_pred.get("sets_under_probability", 0.5)
                    pred_data["sets_pick"] = "Over 3.5" if sets_over > sets_under else "Under 3.5"
                    pred_data["sets_prob"] = max(sets_over, sets_under)
                    pred_data["expected_sets"] = rule_pred.get("expected_sets_played", 0)
                
                # Auto-log this prediction for tracking
                add_prediction({
                    "match_id": f"{p1}_vs_{p2}_{match.get('start_date_lagos', '')}_{match.get('start_time_lagos', '')}",
                    "player1": p1,
                    "player2": p2,
                    "match_date": str(match.get("start_date_lagos", "")),
                    "match_time": str(match.get("start_time_lagos", "")),
                    "location": str(match.get("location", "")),
                    "predicted_winner": predicted_winner,
                    "confidence": winner_prob,
                    "total_pick": pred_data.get("total_pick"),
                    "total_prob": pred_data.get("total_prob"),
                    "first_pick": pred_data.get("first_pick"),
                    "first_prob": pred_data.get("first_prob"),
                    "sets_pick": pred_data.get("sets_pick"),
                    "sets_prob": pred_data.get("sets_prob"),
                })

                all_predictions.append(pred_data)
                
            except Exception:
                continue
    
    # Sort by winner confidence
    all_predictions.sort(key=lambda x: x["winner_prob"], reverse=True)
    
    if not all_predictions:
        st.info(f"🌙 No matches meet the {min_conf_filter}% confidence threshold. Lower the filter to see more.")
        st.stop()
    
    # ===== SUMMARY STATS =====
    st.markdown(f"### 📊 Found {len(all_predictions)} Matches")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    winner_picks = [p for p in all_predictions if p["winner_prob"] >= 0.65]
    with summary_col1:
        st.metric("🎯 Strong Winners", len(winner_picks))
    
    total_picks = [p for p in all_predictions if p.get("total_prob", 0) >= 0.60]
    with summary_col2:
        st.metric("📊 Total Picks", len(total_picks))
    
    first_picks = [p for p in all_predictions if p.get("first_prob", 0) >= 0.60]
    with summary_col3:
        st.metric("⚡ 1st Set Picks", len(first_picks))
    
    sets_picks = [p for p in all_predictions if p.get("sets_prob", 0) >= 0.60]
    with summary_col4:
        st.metric("🎾 Sets Picks", len(sets_picks))
    
    st.markdown("---")
    
    # ===== DISPLAY ALL PICKS =====
    st.markdown("### 🏓 All Matches with Predictions")
    
    for idx, pred in enumerate(all_predictions):
        # Confidence color coding
        conf_pct = pred["winner_prob"] * 100
        if conf_pct >= 70:
            border_color = "#00FF9C"
            conf_emoji = "🔥"
        elif conf_pct >= 60:
            border_color = "#00F5FF"
            conf_emoji = "⭐"
        else:
            border_color = "#FFB800"
            conf_emoji = "⚠️"
        
        # Fair odds & Kelly for winner
        fair_odds = fair_odds_from_probability(pred["winner_prob"])
        market_odds = fair_odds * 0.95
        kelly = kelly_stake(
            current_bankroll,
            pred["winner_prob"],
            market_odds,
            state["kelly_fraction"],
            state["min_stake"],
            state["max_stake_pct"],
        )
        
        # Build the card HTML
        card_html = f"""
        <div class="pick-card" style="border-left: 4px solid {border_color};">
            <div class="pick-title">{conf_emoji} {pred["match"]}</div>
            <div class="pick-time">⏰ {pred["time"]} • {pred["location"]}</div>
        """
        
        # Winner section (always show)
        if "Winner" in market_filter:
            card_html += f"""
            <div style="background: rgba(0, 245, 255, 0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                <div style="color: #B537FF; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">🎯 Winner</div>
                <div style="font-size: 1.3rem; font-weight: 900; color: #00FF9C; margin: 0.3rem 0;">{pred["winner_pred"]}</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width: {conf_pct}%"></div></div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.3rem; font-size: 0.85rem;">
                    <span style="color: #E8ECFF;">Confidence: <strong style="color: #00F5FF;">{conf_pct:.1f}%</strong></span>
                    <span style="color: #E8ECFF;">Odds: <strong style="color: #00F5FF;">{market_odds:.2f}</strong></span>
                    <span style="color: #E8ECFF;">Kelly: <strong style="color: #00F5FF;">{format_currency(kelly['stake'])}</strong></span>
                </div>
            </div>
            """
        
        # Total Points section
        if "Total Points" in market_filter and pred.get("total_pick"):
            total_pct = pred["total_prob"] * 100
            total_odds = fair_odds_from_probability(pred["total_prob"]) * 0.95
            total_kelly = kelly_stake(current_bankroll, pred["total_prob"], total_odds, state["kelly_fraction"], state["min_stake"], state["max_stake_pct"])
            
            card_html += f"""
            <div style="background: rgba(181, 55, 255, 0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                <div style="color: #B537FF; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">📊 Total Points</div>
                <div style="font-size: 1.1rem; font-weight: 900; color: #FFB800; margin: 0.3rem 0;">{pred["total_pick"]}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span style="color: #E8ECFF;">Prob: <strong style="color: #00F5FF;">{total_pct:.1f}%</strong></span>
                    <span style="color: #E8ECFF;">Expected: <strong style="color: #00F5FF;">{pred["expected_total"]:.1f}</strong></span>
                    <span style="color: #E8ECFF;">Kelly: <strong style="color: #00F5FF;">{format_currency(total_kelly['stake'])}</strong></span>
                </div>
            </div>
            """
        
        # First Set Over/Under
        if "First Set O/U" in market_filter and pred.get("first_pick"):
            first_pct = pred["first_prob"] * 100
            first_odds = fair_odds_from_probability(pred["first_prob"]) * 0.95
            first_kelly = kelly_stake(current_bankroll, pred["first_prob"], first_odds, state["kelly_fraction"], state["min_stake"], state["max_stake_pct"])
            
            card_html += f"""
            <div style="background: rgba(0, 255, 156, 0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                <div style="color: #B537FF; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">⚡ First Set 18.5</div>
                <div style="font-size: 1.1rem; font-weight: 900; color: #00FF9C; margin: 0.3rem 0;">{pred["first_pick"]}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span style="color: #E8ECFF;">Prob: <strong style="color: #00F5FF;">{first_pct:.1f}%</strong></span>
                    <span style="color: #E8ECFF;">Expected: <strong style="color: #00F5FF;">{pred["expected_first"]:.1f}</strong></span>
                    <span style="color: #E8ECFF;">Kelly: <strong style="color: #00F5FF;">{format_currency(first_kelly['stake'])}</strong></span>
                </div>
            </div>
            """
        
        # Sets Over/Under
        if "Sets O/U" in market_filter and pred.get("sets_pick"):
            sets_pct = pred["sets_prob"] * 100
            sets_odds = fair_odds_from_probability(pred["sets_prob"]) * 0.95
            sets_kelly = kelly_stake(current_bankroll, pred["sets_prob"], sets_odds, state["kelly_fraction"], state["min_stake"], state["max_stake_pct"])
            
            card_html += f"""
            <div style="background: rgba(255, 184, 0, 0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                <div style="color: #B537FF; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">🎾 Sets 3.5</div>
                <div style="font-size: 1.1rem; font-weight: 900; color: #FFB800; margin: 0.3rem 0;">{pred["sets_pick"]}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span style="color: #E8ECFF;">Prob: <strong style="color: #00F5FF;">{sets_pct:.1f}%</strong></span>
                    <span style="color: #E8ECFF;">Expected: <strong style="color: #00F5FF;">{pred["expected_sets"]:.1f}</strong></span>
                    <span style="color: #E8ECFF;">Kelly: <strong style="color: #00F5FF;">{format_currency(sets_kelly['stake'])}</strong></span>
                </div>
            </div>
            """
        
        # AI reasoning (if ML pred available)
        if pred.get("ml_pred"):
            reason = generate_ai_reasoning(pred["ml_pred"], pred["player1"], pred["player2"])
            card_html += f'<div class="ai-analysis" style="font-size: 0.85rem;">🧠 {reason}</div>'
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Track bet button for the winner
        if kelly['stake'] > 0 and pred["winner_prob"] >= 0.65:
            if st.button(f"📝 Track Winner Bet ({format_currency(kelly['stake'])})", key=f"track_all_{idx}"):
                add_bet(state, {
                    "match": pred["match"],
                    "player1": pred["player1"],
                    "player2": pred["player2"],
                    "prediction": pred["winner_pred"],
                    "confidence": pred["winner_prob"],
                    "stake": kelly['stake'],
                    "odds": market_odds,
                    "market": "match_winner",
                })
                st.success("✅ Tracked in Vault!")
                st.rerun()


# ============================================
# TRACK RECORD PAGE
# ============================================
elif page == "📈 Track Record":
    st.markdown("# 📈 AI TRACK RECORD")
    st.caption("Every prediction, every result - complete transparency")
    
    record = calculate_track_record()
    
    # Overall metrics
    st.markdown("### 📊 Overall Performance")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Predictions", f"{record['total']:,}")
    with m2:
        st.metric("Settled", f"{record['settled']:,}", f"{record['pending']} pending")
    with m3:
        acc = record['winner_accuracy']
        emoji = "🔥" if acc >= 65 else "⭐" if acc >= 60 else "⚠️"
        st.metric("Winner Accuracy", f"{acc:.1f}%", emoji)
    with m4:
        high_acc = record['high_conf_accuracy']
        st.metric("High Confidence", f"{high_acc:.1f}%", f"{record['high_conf_total']} bets")
    
    st.markdown("---")
    
    # Market breakdown
    st.markdown("### 🎯 Accuracy by Market")
    
    markets_data = [
        {"name": "🎯 Winner", "acc": record['winner_accuracy'], "total": record['settled']},
        {"name": "📊 Total Points", "acc": record['total_accuracy'], "total": record.get('total_settled', 0)},
        {"name": "⚡ First Set O/U", "acc": record['first_set_accuracy'], "total": record.get('first_settled', 0)},
        {"name": "🎾 Sets O/U", "acc": record['sets_accuracy'], "total": record.get('sets_settled', 0)},
    ]
    
    for m in markets_data:
        if m['total'] > 0:
            color = "#00FF9C" if m['acc'] >= 60 else "#FFB800" if m['acc'] >= 55 else "#FF3366"
            st.markdown(f"""
            <div class="pick-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #E8ECFF;">{m['name']}</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: {color};">{m['acc']:.1f}%</div>
                </div>
                <div class="confidence-bar" style="margin-top: 0.5rem;">
                    <div class="confidence-fill" style="width: {m['acc']}%; background: {color};"></div>
                </div>
                <div style="color: #B537FF; font-size: 0.85rem; margin-top: 0.3rem;">{m['total']} settled bets</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent predictions
    st.markdown("### 📋 Recent Predictions")
    
    predictions = load_predictions()
    
    if not predictions:
        st.info("No predictions logged yet. Visit the All Picks page to generate predictions.")
    else:
        # Filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            status_filter = st.selectbox("Status", ["All", "Pending", "Settled", "Correct", "Incorrect"])
        with filter_col2:
            limit_show = st.selectbox("Show", [10, 25, 50, 100], index=1)
        
        # Filter predictions
        filtered = predictions
        if status_filter == "Pending":
            filtered = [p for p in filtered if p.get("status") == "pending"]
        elif status_filter == "Settled":
            filtered = [p for p in filtered if p.get("status") == "settled"]
        elif status_filter == "Correct":
            filtered = [p for p in filtered if p.get("winner_correct") == True]
        elif status_filter == "Incorrect":
            filtered = [p for p in filtered if p.get("winner_correct") == False]
        
        # Show most recent first
        filtered = list(reversed(filtered))[:limit_show]
        
        st.caption(f"Showing {len(filtered)} of {len(predictions)} total predictions")
        
        for pred in filtered:
            status = pred.get("status", "pending")
            
            if status == "pending":
                status_emoji = "⏳"
                status_color = "#FFB800"
                result_text = "Awaiting result"
            elif pred.get("winner_correct"):
                status_emoji = "✅"
                status_color = "#00FF9C"
                result_text = f"Correct! Actual: {pred.get('actual_winner', '?')}"
            else:
                status_emoji = "❌"
                status_color = "#FF3366"
                result_text = f"Wrong. Actual: {pred.get('actual_winner', '?')}"
            
            conf = pred.get("confidence", 0) * 100
            
            st.markdown(f"""
            <div class="pick-card" style="border-left: 4px solid {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <div style="font-weight: 700; color: #E8ECFF;">
                        {pred.get('player1', '?')} vs {pred.get('player2', '?')}
                    </div>
                    <div style="font-size: 1.2rem;">{status_emoji}</div>
                </div>
                <div style="color: #B537FF; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    📅 {pred.get('match_date', '?')} • ⏰ {pred.get('match_time', '?')} • 📍 {pred.get('location', '?')}
                </div>
                <div style="background: rgba(0, 245, 255, 0.05); padding: 0.5rem; border-radius: 8px;">
                    <div style="color: #00F5FF; font-weight: 700;">
                        🎯 Predicted: {pred.get('predicted_winner', '?')} ({conf:.1f}%)
                    </div>
                    <div style="color: {status_color}; font-weight: 700; margin-top: 0.3rem;">
                        {result_text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info section
    st.markdown("### 💡 How Track Record Works")
    st.markdown("""
    <div class="pick-card">
        <div style="color: #E8ECFF;">
            <strong>🤖 Automatic Tracking:</strong> Every prediction shown on the All Picks page
            is automatically logged. As matches complete, results are compared against predictions.<br><br>
            <strong>📊 What We Track:</strong><br>
            • Winner predictions (most important)<br>
            • Total points Over/Under 75.5<br>
            • First set Over/Under 18.5<br>
            • Sets Over/Under 3.5<br><br>
            <strong>🎯 Accuracy Goals:</strong><br>
            • 65%+ Winner accuracy → Excellent (green)<br>
            • 60-65% Winner accuracy → Good (cyan)<br>
            • Below 60% → Needs improvement (amber/red)<br><br>
            <em>Higher sample sizes = more reliable accuracy numbers. Focus on the 50+ settled range.</em>
        </div>
    </div>
    """)


# ============================================
# ROADMAP PAGE
# ============================================
elif page == "🗺️ Roadmap":
    st.markdown("# 🗺️ ORACLEBET ROADMAP")
    st.caption("Our journey to becoming the #1 table tennis prediction app")
    
    # Progress overview
    st.markdown("### 📊 Overall Progress")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Completed", "8", "Features")
    with col2:
        st.metric("🚧 In Progress", "3", "Active")
    with col3:
        st.metric("🔮 Planned", "12", "Upcoming")
    
    st.markdown("---")
    
    # PHASE 1: COMPLETED
    st.markdown("## ✅ PHASE 1: FOUNDATION")
    st.caption("What we've already built (Week 1)")
    
    st.markdown("""
    <div class="roadmap-phase roadmap-done">
        <div class="roadmap-status status-done">✅ COMPLETE</div>
        <div class="roadmap-title">🏗️ Core Infrastructure</div>
        <div class="roadmap-features">
            <ul>
                <li>✨ Neon futuristic design system</li>
                <li>🎨 Custom CSS with glow effects</li>
                <li>📱 Mobile-first responsive layout</li>
                <li>🔐 Persistent data storage</li>
                <li>⚙️ User-configurable settings</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-done">
        <div class="roadmap-status status-done">✅ COMPLETE</div>
        <div class="roadmap-title">🤖 ML Prediction Engine</div>
        <div class="roadmap-features">
            <ul>
                <li>📊 XGBoost model trained on 155K+ matches</li>
                <li>🎯 60+ features including fatigue & momentum</li>
                <li>⚡ Dynamic Elo ratings with margin adjustments</li>
                <li>🧠 AI reasoning for each pick</li>
                <li>📈 Model transparency & metrics</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-done">
        <div class="roadmap-status status-done">✅ COMPLETE</div>
        <div class="roadmap-title">💰 Bankroll Management</div>
        <div class="roadmap-features">
            <ul>
                <li>🛡️ Kelly Criterion stake calculator</li>
                <li>📉 Daily loss limit protection</li>
                <li>📋 Complete bet journal</li>
                <li>🔥 Win/loss streak tracking</li>
                <li>💵 Nigerian Naira (₦) support</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PHASE 2: IN PROGRESS
    st.markdown("## 🚧 PHASE 2: ACCURACY BOOST")
    st.caption("What we're working on now (Week 2-3)")
    
    st.markdown("""
    <div class="roadmap-phase roadmap-progress">
        <div class="roadmap-status status-progress">🚧 IN PROGRESS</div>
        <div class="roadmap-title">🎯 Model Accuracy Upgrades</div>
        <div class="roadmap-features">
            <ul>
                <li>⚖️ Time-weighted training (newer matches count more)</li>
                <li>📏 Probability calibration layer</li>
                <li>🎼 Ensemble models (XGBoost + LightGBM + CatBoost)</li>
                <li>🧹 Data cleaning & duplicate player detection</li>
                <li>🎯 Target: 60% → 68% winner accuracy</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-progress">
        <div class="roadmap-status status-progress">🚧 IN PROGRESS</div>
        <div class="roadmap-title">📊 Paper Trading Validation</div>
        <div class="roadmap-features">
            <ul>
                <li>📝 14-day paper trading period</li>
                <li>📈 Real-world accuracy tracking</li>
                <li>💡 Performance analysis dashboard</li>
                <li>🎯 Identify winning bet types</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PHASE 3: UPCOMING
    st.markdown("## 🔮 PHASE 3: NEXT-LEVEL FEATURES")
    st.caption("Coming soon (Month 2)")
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 PLANNED</div>
        <div class="roadmap-title">💬 AI Chat Coach</div>
        <div class="roadmap-features">
            <ul>
                <li>🗨️ Ask questions in plain English</li>
                <li>🎯 "Show me best picks for tonight"</li>
                <li>📊 "Analyze this matchup"</li>
                <li>💰 "How much should I bet?"</li>
                <li>🧠 Powered by GPT/Claude API</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 PLANNED</div>
        <div class="roadmap-title">🎰 Bookmaker Integration</div>
        <div class="roadmap-features">
            <ul>
                <li>📡 Live SportyBet odds scraping</li>
                <li>💎 Value detection (odds vs probability)</li>
                <li>⚖️ Multi-bookmaker comparison</li>
                <li>🎯 Automatic edge alerts</li>
                <li>💰 Best odds finder</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 PLANNED</div>
        <div class="roadmap-title">📺 Live Match Intelligence</div>
        <div class="roadmap-features">
            <ul>
                <li>⚡ Real-time probability updates during matches</li>
                <li>📊 Set-by-set predictions</li>
                <li>🎯 Momentum shift detection</li>
                <li>💰 Cash-out recommendations</li>
                <li>🔔 Live value alerts</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 PLANNED</div>
        <div class="roadmap-title">🧬 Personal Betting DNA</div>
        <div class="roadmap-features">
            <ul>
                <li>📊 Your winning patterns analysis</li>
                <li>⏰ Best times to bet for YOU</li>
                <li>🎯 Player types you profit from</li>
                <li>⚠️ Tilt detection & alerts</li>
                <li>📈 Personalized strategy tips</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PHASE 4: FUTURE
    st.markdown("## 🚀 PHASE 4: ELITE FEATURES")
    st.caption("Long-term vision (Month 3+)")
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 VISION</div>
        <div class="roadmap-title">🌟 Premium Tier</div>
        <div class="roadmap-features">
            <ul>
                <li>👥 Community wisdom + AI hybrid</li>
                <li>📱 Push notifications for hot picks</li>
                <li>🎓 Educational content & tutorials</li>
                <li>🏆 Leaderboard & achievements</li>
                <li>💎 VIP tier with 1-on-1 AI coaching</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="roadmap-phase roadmap-future">
        <div class="roadmap-status status-future">🔮 VISION</div>
        <div class="roadmap-title">📱 Native Mobile App</div>
        <div class="roadmap-features">
            <ul>
                <li>📲 iOS & Android apps</li>
                <li>🔔 Real-time push notifications</li>
                <li>🎨 Native mobile UI</li>
                <li>⚡ Faster performance</li>
                <li>🌐 Offline mode</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Community
    st.markdown("### 💡 Feature Requests")
    st.info("Have an idea? Message us via the app to suggest new features. Your input shapes OracleBet's future!")
    
    st.markdown("### 🎯 Our Mission")
    st.markdown("""
    <div class="pick-card">
        <strong>🔮 OracleBet's Mission:</strong><br><br>
        To provide the most <strong>accurate, transparent, and disciplined</strong> table tennis 
        prediction platform - helping bettors make <strong>data-driven decisions</strong> instead 
        of emotional ones.<br><br>
        <em>We believe that with the right tools, ANYONE can become a profitable bettor. 
        Not through luck, but through discipline, data, and consistency.</em>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🔮 OracleBet v1.0 • Built with 155K+ matches • Paper trade first, bet second")
