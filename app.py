import streamlit as st
import pandas as pd
import numpy as np
from data_acquisition import DataAcquirer
from models import TechnicalModel, SentimentModel, FusionModel
from utils import Visualization, format_large_number
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="BharatSentiment - Indian Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00CC96;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #636EFA;
        border-bottom: 1px solid #2A2A2A;
        padding-bottom: 0.5rem;
    }
    .stock-card {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .positive {
        color: #00CC96;
    }
    .negative {
        color: #EF553B;
    }
    .disclaimer {
        background-color: #2A2A2A;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "data_acquired" not in st.session_state:
    st.session_state.data_acquired = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.markdown('<h1 class="main-header">BharatSentiment</h1>', unsafe_allow_html=True)
st.markdown(
    '<h3 style="text-align: center; color: #636EFA;">Multi-Modal Indian Stock Predictor</h3>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------
st.sidebar.title("Configuration")

stocks_list = [
    "RELIANCE", "TATASTEEL", "INFY", "HDFCBANK", "ICICIBANK",
    "ITC", "SBIN", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK"
]

selected_stock = st.sidebar.selectbox("Select Stock", stocks_list)
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
news_api_key = st.sidebar.text_input("News API Key (Optional)", type="password")
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

st.sidebar.markdown("""
<div class="disclaimer">
    <strong>Important Disclaimer:</strong> This application is for educational and 
    research purposes only. The predictions made by the AI model are not financial advice.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Data Fetch + Analysis
# ------------------------------------------------------------
if analyze_button:
    with st.spinner("Fetching data and analyzing..."):
        acquirer = DataAcquirer()

        # Fetch stock data
        df, pe_ratio, pb_ratio, market_cap = acquirer.get_stock_data(selected_stock, period)

        if df is not None:
            st.session_state.df = df
            st.session_state.pe_ratio = pe_ratio
            st.session_state.pb_ratio = pb_ratio
            st.session_state.market_cap = market_cap
            st.session_state.data_acquired = True

            # Fetch news + sentiment
            headlines, sentiment_scores = acquirer.get_news_sentiment(selected_stock, news_api_key)
            st.session_state.headlines = headlines
            st.session_state.sentiment_scores = sentiment_scores

            # Model initialization
            technical_model = TechnicalModel()
            sentiment_model = SentimentModel()
            fusion_model = FusionModel()

            # Technical model
            X_tech, y_tech = technical_model.prepare_data(df)

            if len(X_tech) > 0:
                # Technical prediction (placeholder)
                tech_prediction = np.mean(y_tech[-10:])

                # Sentiment analysis
                analyzed_sentiment = sentiment_model.analyze_sentiment(headlines)

                # Fundamental data
                fundamental_data = {
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "market_cap": market_cap
                }

                # Fusion features
                fusion_model.prepare_features(
                    tech_prediction,
                    analyzed_sentiment,
                    fundamental_data
                )

                # Bull/Bear logic
                bull_prob = 0.6 if tech_prediction > df["Close"].iloc[-1] else 0.4
                bear_prob = 1 - bull_prob

                # Sentiment adjustment
                positive_sent = sum(1 for s in analyzed_sentiment if s.get("label") == "POSITIVE") / len(
                    analyzed_sentiment)
                bull_prob = min(1.0, bull_prob + (positive_sent * 0.2))
                bear_prob = 1 - bull_prob

                st.session_state.bull_prob = bull_prob
                st.session_state.bear_prob = bear_prob
                st.session_state.analysis_done = True

            else:
                st.error("Insufficient data. Try selecting a longer time period.")
        else:
            st.error("Failed to fetch stock data. Try again later.")

# ------------------------------------------------------------
# Display Results After Data is Loaded
# ------------------------------------------------------------
if st.session_state.data_acquired:
    df = st.session_state.df
    pe_ratio = st.session_state.pe_ratio
    pb_ratio = st.session_state.pb_ratio
    market_cap = st.session_state.market_cap

    # ---------------- Stock Overview ----------------
    st.markdown('<h2 class="sub-header">Stock Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    current_price = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100

    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{change_pct:.2f}%")
    with col2:
        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    with col3:
        st.metric("P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio else "N/A")
    with col4:
        st.metric("Market Cap", format_large_number(market_cap) if market_cap else "N/A")

    # Chart
    st.plotly_chart(Visualization.create_stock_chart(df), use_container_width=True)

    # ---------------- AI Analysis ----------------
    if st.session_state.analysis_done:
        st.markdown('<h2 class="sub-header">AI Analysis Results</h2>', unsafe_allow_html=True)

        bull_prob = st.session_state.bull_prob
        bear_prob = st.session_state.bear_prob
        headlines = st.session_state.headlines
        sentiment_scores = st.session_state.sentiment_scores

        prediction = "Bullish" if bull_prob > bear_prob else "Bearish"
        confidence = (bull_prob if prediction == "Bullish" else bear_prob) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                Visualization.create_prediction_gauge(prediction, confidence),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                Visualization.create_sentiment_chart(sentiment_scores),
                use_container_width=True
            )

            st.info(f"""
            **Analysis Summary:**
            - **Technical Indicators:** {'Bullish' if bull_prob > 0.6 else 'Bearish' if bear_prob > 0.6 else 'Neutral'}
            - **News Sentiment:** {sum(1 for s in sentiment_scores if s.get('label') == 'POSITIVE') / len(sentiment_scores) * 100:.1f}% Positive
            - **Fundamentals:** {'Strong' if pe_ratio and pe_ratio < 25 else 'Average' if pe_ratio and pe_ratio < 40 else 'Weak' if pe_ratio else 'N/A'}
            """)

        # ---------------- News Headlines ----------------
        st.markdown('<h3 class="sub-header">Recent News Sentiment</h3>', unsafe_allow_html=True)

        for i, (headline, sentiment) in enumerate(zip(headlines, sentiment_scores)):
            label = sentiment.get("label", "NEUTRAL")
            score = sentiment.get("score", 0.5)

            icon = "🟢" if label == "POSITIVE" else "🔴" if label == "NEGATIVE" else "🔵"
            cls = "positive" if label == "POSITIVE" else "negative" if label == "NEGATIVE" else ""

            st.markdown(f"""
            <div class="stock-card">
                <p>{icon} <span class="{cls}">{headline}</span> 
                (Confidence: {score:.2f})</p>
            </div>
            """, unsafe_allow_html=True)

            if i >= 4:
                break

# ------------------------------------------------------------
# Initial App Message
# ------------------------------------------------------------
if not st.session_state.data_acquired:
    st.info("""
    Welcome to **BharatSentiment** — A multi-modal Indian stock analysis system.

    **This app analyzes stocks using:**
    - 📈 Technical Indicators  
    - 📰 News Sentiment  
    - 🧮 Fundamentals  

    **Get Started:**
    1. Select a stock  
    2. Choose time period  
    3. Click **Analyze Stock**  
    """)

    sample_df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, 101, 105, 107],
        "Volume": [1_000_000, 1_200_000, 800_000, 1_500_000, 2_000_000],
        "MA20": [100, 100.5, 101, 101.5, 102],
        "RSI": [45, 50, 55, 60, 65],
        "MA50": [99, 99.5, 100, 100.5, 101],
        "MACD": [0.1, 0.2, 0.3, 0.4, 0.5],
        "MACD_Signal": [0.05, 0.15, 0.25, 0.35, 0.45],
        "MACD_Histogram": [0.05] * 5,
        "BB_Middle": [100, 101, 102, 103, 104]
    })

    st.plotly_chart(Visualization.create_stock_chart(sample_df), use_container_width=True)
