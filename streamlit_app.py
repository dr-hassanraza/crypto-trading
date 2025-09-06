import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Any

# Import our modules (with error handling for Streamlit deployment)
try:
    from src.ml_models.price_prediction import ml_predictor
    from src.sentiment.social_sentiment import social_sentiment_analyzer
    from src.arbitrage.arbitrage_detector import arbitrage_detector
    from src.defi.defi_analyzer import defi_analyzer
    from src.onchain.whale_tracker import whale_tracker
    from src.risk.advanced_risk_models import risk_models
    ml_modules_available = True
except ImportError as e:
    st.warning(f"Some ML modules not available in Streamlit Cloud: {e}")
    ml_modules_available = False

# Page configuration
st.set_page_config(
    page_title="Crypto Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .status-operational {
        color: #28a745;
    }
    .status-warning {
        color: #ffc107;
    }
    .status-error {
        color: #dc3545;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Crypto Trend Analyzer</h1>
    <p>Enterprise-Grade Cryptocurrency Analysis Platform</p>
    <small>Powered by Advanced AI & Machine Learning</small>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Dashboard Controls")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Select Dashboard",
    ["📊 Overview", "🤖 AI Predictions", "💹 Trading Analytics", "⚠️ Risk Management", "🏛️ DeFi Analysis", "🐋 Whale Tracker"]
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    st.sidebar.info(f"Data refreshes every {refresh_interval} seconds")

# Mock data generation functions
def generate_mock_price_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate realistic mock price data."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = {'BTC': 45000, 'ETH': 2500, 'ADA': 0.5, 'SOL': 100}.get(symbol, 100)
    
    returns = np.random.normal(0.001, 0.03, days)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        volatility = abs(returns[i]) * 2 + 0.01
        open_price = prices[i-1] if i > 0 else price
        high = max(open_price, price) * (1 + volatility/2)
        low = min(open_price, price) * (1 - volatility/2)
        volume = np.random.uniform(1000000, 10000000)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(price, 2),
            'Volume': round(volume)
        })
    
    return pd.DataFrame(data)

def get_mock_predictions():
    """Get mock ML predictions."""
    return {
        'BTC': {
            'short_term_prediction': {
                'direction': np.random.choice(['up', 'down', 'sideways']),
                'confidence': np.random.uniform(0.6, 0.95),
                'predicted_price': 45000 + np.random.normal(0, 2000)
            }
        },
        'ETH': {
            'short_term_prediction': {
                'direction': np.random.choice(['up', 'down', 'sideways']),
                'confidence': np.random.uniform(0.6, 0.95),
                'predicted_price': 2500 + np.random.normal(0, 200)
            }
        },
        'ADA': {
            'short_term_prediction': {
                'direction': np.random.choice(['up', 'down', 'sideways']),
                'confidence': np.random.uniform(0.6, 0.95),
                'predicted_price': 0.5 + np.random.normal(0, 0.05)
            }
        }
    }

def get_mock_arbitrage_opportunities():
    """Get mock arbitrage opportunities."""
    opportunities = []
    symbols = ['BTC', 'ETH', 'ADA']
    exchanges = ['Binance', 'Coinbase', 'Kraken', 'KuCoin']
    
    for _ in range(np.random.randint(2, 6)):
        symbol = np.random.choice(symbols)
        buy_exchange = np.random.choice(exchanges)
        sell_exchange = np.random.choice([e for e in exchanges if e != buy_exchange])
        
        base_price = {'BTC': 45000, 'ETH': 2500, 'ADA': 0.5}[symbol]
        buy_price = base_price * (1 - np.random.uniform(0.001, 0.01))
        sell_price = base_price * (1 + np.random.uniform(0.001, 0.01))
        
        opportunities.append({
            'symbol': symbol,
            'buy_exchange': buy_exchange,
            'sell_exchange': sell_exchange,
            'buy_price': round(buy_price, 2),
            'sell_price': round(sell_price, 2),
            'profit_percentage': (sell_price - buy_price) / buy_price,
            'estimated_profit': round((sell_price - buy_price) * 10, 2)
        })
    
    return {'simple_arbitrage': opportunities}

# Main content based on selected page
if page == "📊 Overview":
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_value = 300000 + np.random.normal(0, 10000)
        st.metric("Portfolio Value", f"${portfolio_value:,.0f}", f"{np.random.uniform(-5, 5):.2f}%")
    
    with col2:
        daily_pnl = np.random.normal(500, 2000)
        st.metric("Daily P&L", f"${daily_pnl:,.0f}", f"{np.random.uniform(-3, 3):.2f}%")
    
    with col3:
        sentiment_score = np.random.uniform(30, 85)
        st.metric("Market Sentiment", f"{sentiment_score:.1f}/100", "↑ Bullish")
    
    with col4:
        active_alerts = np.random.randint(0, 8)
        st.metric("Active Alerts", f"{active_alerts}", "🔔")
    
    # Price charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BTC/USD Price Chart")
        btc_data = generate_mock_price_data('BTC', 30)
        fig = go.Figure(data=go.Candlestick(
            x=btc_data['Date'],
            open=btc_data['Open'],
            high=btc_data['High'],
            low=btc_data['Low'],
            close=btc_data['Close']
        ))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ETH/USD Price Chart")
        eth_data = generate_mock_price_data('ETH', 30)
        fig = go.Figure(data=go.Candlestick(
            x=eth_data['Date'],
            open=eth_data['Open'],
            high=eth_data['High'],
            low=eth_data['Low'],
            close=eth_data['Close']
        ))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Current prices
    st.subheader("Live Market Data")
    price_col1, price_col2, price_col3, price_col4 = st.columns(4)
    
    with price_col1:
        btc_price = 45000 + np.random.normal(0, 1000)
        st.metric("BTC", f"${btc_price:,.0f}", f"{np.random.uniform(-5, 5):.2f}%")
    
    with price_col2:
        eth_price = 2500 + np.random.normal(0, 100)
        st.metric("ETH", f"${eth_price:,.0f}", f"{np.random.uniform(-5, 5):.2f}%")
    
    with price_col3:
        ada_price = 0.5 + np.random.normal(0, 0.05)
        st.metric("ADA", f"${ada_price:.4f}", f"{np.random.uniform(-5, 5):.2f}%")
    
    with price_col4:
        sol_price = 100 + np.random.normal(0, 10)
        st.metric("SOL", f"${sol_price:.2f}", f"{np.random.uniform(-5, 5):.2f}%")

elif page == "🤖 AI Predictions":
    st.header("AI Price Predictions")
    
    predictions = get_mock_predictions()
    
    for symbol, pred in predictions.items():
        st.subheader(f"{symbol} Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction = pred['short_term_prediction']['direction']
            direction_color = {'up': '🟢', 'down': '🔴', 'sideways': '🟡'}[direction]
            st.metric("Direction", f"{direction_color} {direction.upper()}")
        
        with col2:
            confidence = pred['short_term_prediction']['confidence']
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        with col3:
            predicted_price = pred['short_term_prediction']['predicted_price']
            st.metric("Predicted Price", f"${predicted_price:,.2f}")
        
        # Prediction chart
        dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
        current_price = {'BTC': 45000, 'ETH': 2500, 'ADA': 0.5}[symbol]
        
        # Generate prediction path
        if direction == 'up':
            price_path = np.linspace(current_price, predicted_price * 1.05, 7)
        elif direction == 'down':
            price_path = np.linspace(current_price, predicted_price * 0.95, 7)
        else:
            price_path = np.full(7, current_price) + np.random.normal(0, current_price * 0.01, 7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=price_path, mode='lines+markers', name=f'{symbol} Prediction'))
        fig.update_layout(title=f"{symbol} 7-Day Prediction", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

elif page == "💹 Trading Analytics":
    st.header("Trading Analytics")
    
    # Arbitrage opportunities
    st.subheader("🔍 Arbitrage Opportunities")
    opportunities = get_mock_arbitrage_opportunities()
    
    if opportunities['simple_arbitrage']:
        arb_df = pd.DataFrame(opportunities['simple_arbitrage'])
        arb_df['Profit %'] = (arb_df['profit_percentage'] * 100).round(2)
        
        st.dataframe(
            arb_df[['symbol', 'buy_exchange', 'sell_exchange', 'buy_price', 'sell_price', 'Profit %', 'estimated_profit']],
            column_config={
                "symbol": "Symbol",
                "buy_exchange": "Buy Exchange",
                "sell_exchange": "Sell Exchange", 
                "buy_price": st.column_config.NumberColumn("Buy Price", format="$%.2f"),
                "sell_price": st.column_config.NumberColumn("Sell Price", format="$%.2f"),
                "Profit %": st.column_config.NumberColumn("Profit %", format="%.2f%%"),
                "estimated_profit": st.column_config.NumberColumn("Est. Profit", format="$%.2f")
            },
            use_container_width=True
        )
    else:
        st.info("No arbitrage opportunities found at the moment.")
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    volume_data = {
        'Exchange': ['Binance', 'Coinbase', 'Kraken', 'KuCoin'],
        'BTC Volume': np.random.uniform(10000, 50000, 4),
        'ETH Volume': np.random.uniform(50000, 200000, 4),
        'Total Volume (24h)': np.random.uniform(1000000, 5000000, 4)
    }
    
    volume_df = pd.DataFrame(volume_data)
    fig = px.bar(volume_df, x='Exchange', y='Total Volume (24h)', title="24h Trading Volume by Exchange")
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Risk Management":
    st.header("Risk Management Dashboard")
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_score = np.random.uniform(3, 8)
        st.metric("Risk Score", f"{risk_score:.1f}/10", "🔴 High" if risk_score > 7 else "🟡 Medium" if risk_score > 4 else "🟢 Low")
    
    with col2:
        portfolio_var = np.random.uniform(2, 8)
        st.metric("Portfolio VaR (95%)", f"{portfolio_var:.2f}%", "📉")
    
    with col3:
        max_drawdown = np.random.uniform(5, 15)
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%", "📊")
    
    # Risk distribution
    st.subheader("Portfolio Risk Distribution")
    risk_data = {
        'Asset': ['BTC', 'ETH', 'ADA', 'SOL'],
        'Allocation %': [40, 30, 20, 10],
        'Risk Contribution': np.random.uniform(15, 35, 4),
        'Volatility': np.random.uniform(30, 80, 4)
    }
    
    risk_df = pd.DataFrame(risk_data)
    
    fig = px.pie(risk_df, values='Allocation %', names='Asset', title="Portfolio Allocation")
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress test results
    st.subheader("Stress Test Results")
    stress_scenarios = {
        'Scenario': ['Market Crash (-30%)', 'Flash Crash (-50%)', 'Regulatory Ban', 'Exchange Hack', 'Liquidity Crisis'],
        'Portfolio Impact': ['-25.2%', '-42.1%', '-18.7%', '-8.3%', '-15.4%'],
        'Recovery Time': ['6 months', '12 months', '3 months', '1 month', '4 months']
    }
    
    stress_df = pd.DataFrame(stress_scenarios)
    st.dataframe(stress_df, use_container_width=True)

elif page == "🏛️ DeFi Analysis":
    st.header("DeFi Yield Analysis")
    
    # DeFi yields
    defi_data = {
        'Protocol': ['Compound', 'Aave', 'Yearn Finance', 'Curve', 'Uniswap V3'],
        'Asset': ['USDC', 'ETH', 'DAI', 'USDT', 'ETH-USDC'],
        'APY %': np.random.uniform(3, 25, 5),
        'TVL ($M)': np.random.uniform(100, 2000, 5),
        'Risk Level': np.random.choice(['Low', 'Medium', 'High'], 5)
    }
    
    defi_df = pd.DataFrame(defi_data)
    defi_df['APY %'] = defi_df['APY %'].round(2)
    defi_df['TVL ($M)'] = defi_df['TVL ($M)'].round(0)
    
    st.dataframe(
        defi_df,
        column_config={
            "APY %": st.column_config.NumberColumn("APY %", format="%.2f%%"),
            "TVL ($M)": st.column_config.NumberColumn("TVL", format="$%.0fM"),
            "Risk Level": st.column_config.SelectboxColumn("Risk", options=['Low', 'Medium', 'High'])
        },
        use_container_width=True
    )
    
    # Yield comparison chart
    fig = px.scatter(defi_df, x='TVL ($M)', y='APY %', color='Risk Level', 
                     size='TVL ($M)', hover_data=['Protocol'], 
                     title="DeFi Yield vs TVL Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Impermanent loss calculator
    st.subheader("Impermanent Loss Calculator")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_price_ratio = st.number_input("Initial Price Ratio", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        current_price_ratio = st.number_input("Current Price Ratio", value=1.2, min_value=0.1, max_value=10.0, step=0.1)
    
    with col2:
        if current_price_ratio != initial_price_ratio:
            ratio = current_price_ratio / initial_price_ratio
            il = 2 * np.sqrt(ratio) / (1 + ratio) - 1
            st.metric("Impermanent Loss", f"{il*100:.2f}%", "📉" if il < 0 else "📈")
        else:
            st.metric("Impermanent Loss", "0.00%", "📊")

elif page == "🐋 Whale Tracker":
    st.header("Whale Activity Tracker")
    
    # Large transactions
    whale_data = {
        'Timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
        'Symbol': np.random.choice(['BTC', 'ETH', 'ADA'], 10),
        'Amount': np.random.uniform(1000000, 50000000, 10),
        'From Exchange': np.random.choice(['Unknown', 'Binance', 'Coinbase', 'Kraken'], 10),
        'To Exchange': np.random.choice(['Unknown', 'Binance', 'Coinbase', 'Kraken'], 10),
        'Type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], 10)
    }
    
    whale_df = pd.DataFrame(whale_data)
    whale_df['Amount'] = whale_df['Amount'].round(0)
    
    st.dataframe(
        whale_df,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Time"),
            "Amount": st.column_config.NumberColumn("Amount", format="$%.0f")
        },
        use_container_width=True
    )
    
    # Whale activity chart
    daily_whale_activity = whale_df.groupby(whale_df['Timestamp'].dt.date)['Amount'].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_whale_activity.index, y=daily_whale_activity.values))
    fig.update_layout(title="Daily Whale Activity Volume", xaxis_title="Date", yaxis_title="Volume ($)")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 Crypto Trend Analyzer | Enterprise-Grade Analytics Platform</p>
    <p><small>Powered by Advanced AI & Machine Learning | Real-time Market Analysis</small></p>
    <p><small>⚠️ This is a demo application with simulated data for educational purposes</small></p>
</div>
""", unsafe_allow_html=True)

# Auto refresh functionality
if auto_refresh:
    import time
    import threading
    
    # Simple auto-refresh mechanism
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()