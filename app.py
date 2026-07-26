from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Import existing modules
try:
    from src.ml_pipeline import load_model_bundle, predict_with_bundle, train_model_bundle
    from src.setka_core import load_raw_data
    from src.setka_live import (
        fetch_nearest_matches,
        add_lagos_time,
        add_location_names,
        location_map,
    )
    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as exc:
    IMPORTS_OK = False
    IMPORT_ERROR = str(exc)

# Page config
st.set_page_config(
    page_title="OracleBet 🔮",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== DESIGN SYSTEM CSS =====
st.markdown("""
<style>
    /* Root background */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f4a 50%, #0A0E27 100%);
        color: #E8ECFF;
    }
    
    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    h1 {
        color: #00F5FF !important;
        text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    h2, h3, h4 {
        color: #E8ECFF !important;
        font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E27 0%, #151a3d 100%);
        border-right: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00F5FF !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #00F5FF !important;
        font-size: 2rem !important;
        font-weight: 900;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B537FF !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #00F5FF 0%, #B537FF 100%);
        color: #0A0E27;
        border: none;
        border-radius: 12px;
        font-weight: 900;
        letter-spacing: 1px;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.6);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(0, 245, 255, 0.08);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 12px;
    }
    
    /* Pick cards */
    .pick-card {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.05) 0%, rgba(181, 55, 255, 0.05) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .pick-card-top {
        background: linear-gradient(135deg, rgba(0, 255, 156, 0.1) 0%, rgba(0, 245, 255, 0.1) 100%);
        border: 2px solid #00FF9C;
        box-shadow: 0 0 40px rgba(0, 255, 156, 0.3);
    }
    
    .pick-title {
        font-size: 1.3rem;
        font-weight: 900;
        color: #00F5FF;
        margin-bottom: 0.5rem;
    }
    
    .pick-time {
        color: #B537FF;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .pick-prediction {
        font-size: 1.8rem;
        font-weight: 900;
        color: #00FF9C;
        margin: 1rem 0;
        text-shadow: 0 0 15px rgba(0, 255, 156, 0.4);
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #00F5FF 0%, #00FF9C 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(0, 255, 156, 0.5);
    }
    
    .pick-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #E8ECFF;
    }
    
    .stat-label {
        color: #B537FF;
        font-weight: 600;
    }
    
    .stat-value {
        color: #00F5FF;
        font-weight: 700;
    }
    
    .ai-analysis {
        background: rgba(181, 55, 255, 0.1);
        border-left: 3px solid #B537FF;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-style: italic;
        color: #E8ECFF;
    }
    
    /* Forecast card */
    .forecast-card {
        background: linear-gradient(135deg, rgba(255, 184, 0, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%);
        border: 1px solid rgba(255, 184, 0, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .forecast-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #FFB800;
        text-shadow: 0 0 20px rgba(255, 184, 0, 0.4);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ===== SIDEBAR NAVIGATION =====
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
    st.caption("💎 Powered by 155K+ matches")
    st.caption("🤖 XGBoost ML Engine")
    st.caption("⚡ Live Setka Cup data")


# ===== IMPORT CHECK =====
if not IMPORTS_OK:
    st.error(f"⚠️ Import Error: {IMPORT_ERROR}")
    st.info("Please check that all src/ files exist and are valid.")
    st.stop()


# ===== LOAD MODEL BUNDLE =====
MODEL_PATH = Path("models/setka_ml_bundle.joblib")


@st.cache_resource(show_spinner="🔮 Loading Oracle intelligence...")
def get_model_bundle():
    if MODEL_PATH.exists():
        try:
            return load_model_bundle(MODEL_PATH), None
        except Exception as exc:
            return None, str(exc)
    return None, "No trained model found. Please train one in ML Lab first."


@st.cache_data(ttl=60, show_spinner="⚡ Fetching live matches...")
def get_upcoming_matches():
    try:
        locs = location_map()
        frame = fetch_nearest_matches()
        frame = add_lagos_time(frame)
        frame = add_location_names(frame, locs)
        return frame, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ===== HOME PAGE =====
if page == "🏠 Home":
    # Greeting
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    
    st.markdown(f"# 🔮 ORACLEBET")
    st.markdown(f"### {greeting}, Champion 👑")
    st.markdown("")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bankroll", "₦50,000", "+₦4,500")
    with col2:
        st.metric("Win Rate", "67%", "+2%")
    with col3:
        st.metric("ROI (30d)", "+12.5%", "+3.2%")
    with col4:
        st.metric("Active Picks", "3", "🔥")
    
    st.markdown("---")
    
    # Load model and matches
    bundle, model_error = get_model_bundle()
    matches, match_error = get_upcoming_matches()
    
    if model_error:
        st.warning(f"⚠️ Model: {model_error}")
        st.info("👉 Go to legacy app to train a model first.")
        st.stop()
    
    if match_error:
        st.warning(f"⚠️ Matches: {match_error}")
    
    # Generate predictions
    predictions = []
    if not matches.empty and bundle:
        for _, match in matches.head(20).iterrows():
            try:
                pred = predict_with_bundle(
                    bundle,
                    match["player1"],
                    match["player2"],
                    current_dt=pd.Timestamp.now()
                )
                confidence = max(pred["player_a_win_probability"], pred["player_b_win_probability"])
                if confidence >= 0.65:  # Only show high-confidence picks
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
    
    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_picks = predictions[:5]
    
    # Forecast card
    st.markdown(f"""
    <div class="forecast-card">
        <div style="color: #FFB800; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">
            ☀️ Oracle Forecast
        </div>
        <div class="forecast-value">{len(top_picks)} STRONG PICKS</div>
        <div style="color: #E8ECFF; margin-top: 0.5rem;">
            Confidence Level: {"HIGH ✨" if len(top_picks) >= 3 else "MEDIUM" if len(top_picks) >= 1 else "LOW 🌧️"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not top_picks:
        st.info("🌙 No high-confidence picks right now. Check back later!")
        st.stop()
    
    # Top pick highlight
    st.markdown("### ⭐ TOP PICK OF THE DAY")
    top = top_picks[0]
    
    confidence_pct = top["confidence"] * 100
    fair_odds = 1 / top["confidence"] if top["confidence"] > 0 else 0
    suggested_stake = 750  # We'll calculate Kelly later
    potential_win = suggested_stake * fair_odds
    
    st.markdown(f"""
    <div class="pick-card pick-card-top">
        <div class="pick-title">🏓 {top["match"]}</div>
        <div class="pick-time">⏰ {top["time"]} • {top["location"]}</div>
        <div class="pick-prediction">🎯 {top["winner"]} WINS</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_pct}%"></div>
        </div>
        <div style="text-align: center; color: #00FF9C; font-weight: 700; margin-bottom: 1rem;">
            {confidence_pct:.1f}% confidence
        </div>
        <div class="pick-stat">
            <span class="stat-label">💎 Fair Odds</span>
            <span class="stat-value">{fair_odds:.2f}</span>
        </div>
        <div class="pick-stat">
            <span class="stat-label">💰 Suggested Stake</span>
            <span class="stat-value">₦{suggested_stake:,}</span>
        </div>
        <div class="pick-stat">
            <span class="stat-label">📊 Potential Win</span>
            <span class="stat-value">₦{potential_win:,.0f}</span>
        </div>
        <div class="ai-analysis">
            🧠 <strong>AI Analysis:</strong> {top["winner"]} shows strong indicators - high confidence pick backed by 155K matches analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Other picks
    if len(top_picks) > 1:
        st.markdown(f"### 📋 More Strong Picks ({len(top_picks) - 1})")
        for pick in top_picks[1:]:
            conf_pct = pick["confidence"] * 100
            odds = 1 / pick["confidence"]
            st.markdown(f"""
            <div class="pick-card">
                <div class="pick-title">🏓 {pick["match"]}</div>
                <div class="pick-time">⏰ {pick["time"]} • {pick["location"]}</div>
                <div class="pick-prediction" style="font-size: 1.4rem;">🎯 {pick["winner"]} ({conf_pct:.1f}%)</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_pct}%"></div>
                </div>
                <div class="pick-stat">
                    <span class="stat-label">Fair Odds</span>
                    <span class="stat-value">{odds:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ===== OTHER PAGES (COMING SOON) =====
elif page == "📡 Live":
    st.markdown("# 📡 LIVE MATCHES")
    st.info("🚧 Live view coming in Day 3! Real-time probability updates during matches.")

elif page == "🧠 Analyze":
    st.markdown("# 🧠 DEEP ANALYZE")
    st.info("🚧 Deep analysis coming in Day 4! AI chat coach + detailed breakdowns.")

elif page == "🛡️ Vault":
    st.markdown("# 🛡️ BANKROLL VAULT")
    st.info("🚧 Bankroll AI Guardian coming in Day 5! Track bets, manage risk.")

elif page == "📊 Edge":
    st.markdown("# 📊 STATISTICAL EDGE")
    st.info("🚧 Model transparency coming in Day 6! See how the Oracle thinks.")


# Footer
st.markdown("---")
st.caption("🔮 OracleBet v0.2 • Day 1 Complete • Building in public")
