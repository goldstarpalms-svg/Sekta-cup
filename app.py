import streamlit as st

st.set_page_config(
    page_title="OracleBet - Coming Soon",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the coming soon page
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f4a 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f4a 100%);
    }
    h1 {
        color: #00F5FF !important;
        text-align: center;
        font-size: 4rem !important;
        text-shadow: 0 0 20px #00F5FF;
    }
    h2, h3 {
        color: #E8ECFF !important;
        text-align: center;
    }
    .stMarkdown p {
        color: #E8ECFF;
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("# 🔮 ORACLEBET")
st.markdown("### The Future of Table Tennis Predictions")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.info("🚧 **Building something legendary...**")
    st.markdown("**Powered by:**")
    st.markdown("- 🧠 155,000+ historical matches")
    st.markdown("- 🤖 XGBoost ML models") 
    st.markdown("- ⚡ Real-time Setka Cup data")
    st.markdown("- 💎 AI-powered edge detection")
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("**Day 1 of 7 - Foundation Complete ✅**")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("<p style='text-align: center; color: #B537FF;'>OracleBet v0.1 • Building in public</p>", unsafe_allow_html=True)
