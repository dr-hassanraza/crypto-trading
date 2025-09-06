# 🚀 Advanced Crypto Trend Analyzer AI Agent

A comprehensive, AI-powered cryptocurrency analysis system with real-time data streaming, advanced technical analysis, portfolio management, and automated trading signals.

## 🌟 Features

### 🤖 AI-Powered Analysis
- **OpenAI GPT-4 Integration**: Advanced market analysis with natural language insights
- **Multi-timeframe Analysis**: 1h, 4h, 1d, 1w technical analysis
- **Sentiment Analysis**: News and social media sentiment integration
- **Pattern Recognition**: Automated detection of chart patterns and signals

### 📊 Advanced Technical Indicators
- **Momentum**: RSI, Stochastic, Rate of Change
- **Trend**: MACD, ADX, Moving Averages (SMA, EMA)
- **Volatility**: Bollinger Bands, ATR, Volatility measures
- **Volume**: OBV, Volume SMA, Volume ratios
- **Support/Resistance**: Pivot points, dynamic levels

### 🔄 Real-Time Data Streaming
- **WebSocket Connections**: Live price feeds from Binance
- **Multi-Source Data**: CoinGecko, Binance, News APIs
- **Anomaly Detection**: Real-time price anomaly alerts
- **Performance Monitoring**: Stream health and latency tracking

### 💼 Portfolio Management
- **Risk Assessment**: Comprehensive risk scoring and limits
- **Position Sizing**: Intelligent position sizing with Kelly Criterion
- **Diversification**: Automatic diversification scoring
- **Stop-Loss/Take-Profit**: Dynamic risk management levels
- **Performance Tracking**: Detailed P&L and metrics tracking

### 📈 Signal Generation
- **Confidence Scoring**: 0-100% confidence levels for all signals
- **Multi-Factor Analysis**: Technical + AI + Sentiment combined
- **Risk-Adjusted Signals**: Position sizing based on risk assessment
- **Signal History**: Track signal performance and accuracy

### 🧪 Backtesting Engine
- **Historical Testing**: Validate strategies on historical data
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, etc.
- **Strategy Comparison**: Compare multiple strategies side-by-side
- **Detailed Reports**: Comprehensive backtesting reports

### 🌐 Web Dashboard
- **Real-Time Updates**: Live price charts and signal updates
- **Interactive Charts**: Plotly-powered technical analysis charts
- **Portfolio Overview**: Real-time portfolio tracking
- **Alert Management**: System health and trading alerts

### 📝 Comprehensive Logging & Monitoring
- **Structured Logging**: JSON-formatted logs with rotation
- **System Health**: CPU, memory, disk, network monitoring
- **Performance Metrics**: API calls, error rates, response times
- **Alert System**: Email notifications for critical events

## 🏗️ Architecture

```
crypto-trend-analyzer/
├── main.py                     # Main application entry point
├── config/
│   └── config.py              # Configuration management
├── src/
│   ├── analyzers/
│   │   ├── technical_indicators.py  # Technical analysis
│   │   └── ai_analysis.py          # AI-powered analysis
│   ├── data_sources/
│   │   ├── market_data.py          # Multi-source data fetcher
│   │   └── realtime_streams.py     # WebSocket streaming
│   ├── signals/
│   │   └── signal_generator.py     # Advanced signal generation
│   ├── portfolio/
│   │   └── portfolio_manager.py    # Portfolio management
│   ├── backtesting/
│   │   └── backtest_engine.py      # Backtesting system
│   ├── dashboard/
│   │   └── main.py                 # FastAPI web dashboard
│   └── utils/
│       ├── logging_config.py       # Logging system
│       └── monitoring.py           # System monitoring
├── logs/                      # Log files directory
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Crypto-Trend-Analyzer-AI-Agent-main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
# OpenAI Configuration (Required for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Binance API (Optional - for real-time data)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# News API (Optional - for sentiment analysis)
NEWS_API_KEY=your_news_api_key

# Email Configuration (Optional - for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
TO_EMAIL=recipient@gmail.com
```

### 3. Run the System

Choose one of the following options:

```bash
# Run analysis engine only
python main.py analyze

# Run web dashboard only
python main.py dashboard

# Run backtesting
python main.py backtest

# Run everything (recommended)
python main.py all
```

### 4. Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📊 Usage Examples

### Signal Generation

The system automatically generates trading signals based on:

```python
# Example signal output
{
    "symbol": "BTCUSDT",
    "signal": "BUY",
    "confidence": 78.5,
    "strength": 0.65,
    "entry_price": 45000.00,
    "stop_loss": 42750.00,
    "take_profit": 49500.00,
    "risk_reward_ratio": "1:2.33",
    "time_horizon": "1-2 weeks"
}
```

### Portfolio Management

```python
# Portfolio metrics
{
    "total_value": 12500.50,
    "total_return": 25.01,
    "max_drawdown": -8.3,
    "sharpe_ratio": 1.87,
    "win_rate": 72.3,
    "positions_count": 5,
    "risk_level": "MODERATE"
}
```

### Backtesting Results

```python
# Backtest performance
{
    "total_return": 34.7,
    "annualized_return": 28.9,
    "sharpe_ratio": 2.1,
    "max_drawdown": -12.4,
    "win_rate": 68.2,
    "total_trades": 47
}
```

## 🔧 Configuration

### Trading Parameters

```python
# config/config.py
class Config:
    # Trading settings
    CRYPTOS = ['bitcoin', 'ethereum', 'solana', 'cardano']
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    
    # Risk management
    MAX_POSITION_SIZE = 0.1     # 10% max per position
    STOP_LOSS_PCT = 0.05        # 5% stop loss
    TAKE_PROFIT_PCT = 0.15      # 15% take profit
    
    # Signal thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MIN_CONFIDENCE = 60         # Minimum signal confidence
```

### Monitoring Thresholds

```python
# System monitoring alerts
THRESHOLDS = {
    'cpu_usage': 80.0,          # CPU usage %
    'memory_usage': 85.0,       # Memory usage %
    'api_error_rate': 0.1,      # 10% error rate
    'max_drawdown': 0.20        # 20% max drawdown
}
```

## 📈 Signal Types

### Buy Signals
- **STRONG_BUY**: High confidence (>80%), strong technical + AI alignment
- **BUY**: Medium confidence (60-80%), positive technical indicators

### Sell Signals
- **STRONG_SELL**: High confidence (>80%), strong bearish alignment
- **SELL**: Medium confidence (60-80%), negative technical indicators

### Hold Signals
- **HOLD**: Low confidence (<60%) or conflicting signals

## 🛡️ Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate and average returns
- **Risk Limits**: Maximum 15% per position, 40% per sector
- **Cash Management**: Minimum 10% cash reserves

### Stop Loss Strategy
- **Dynamic Stops**: Adjusted based on volatility and signal strength
- **Technical Stops**: Based on support/resistance levels
- **Time Stops**: Maximum holding period limits

## 📊 Performance Metrics

### Portfolio Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### System Metrics
- **Signal Accuracy**: Historical signal performance
- **API Response Times**: Data source performance
- **System Uptime**: Service availability
- **Error Rates**: System reliability metrics

## 🔍 Monitoring & Alerts

### Health Checks
- **Market Data**: Data source connectivity and freshness
- **Real-time Streams**: WebSocket connection status
- **AI Analysis**: OpenAI API availability
- **Portfolio**: Position and risk status

### Alert Types
- **System Alerts**: High CPU/memory usage, API failures
- **Trading Alerts**: High-confidence signals, stop losses triggered
- **Risk Alerts**: Position size limits, drawdown warnings
- **Market Alerts**: Price anomalies, volatility spikes

## 🧪 Testing & Validation

### Backtesting
```bash
# Run comprehensive backtest
python main.py backtest

# Results include:
# - Total return and risk metrics
# - Trade-by-trade analysis
# - Equity curve visualization
# - Strategy performance comparison
```

### Paper Trading
- Real-time signal generation without actual trades
- Portfolio simulation with real market data
- Performance tracking and analysis

## 🔒 Security

### API Keys
- Store sensitive keys in `.env` file
- Use environment variables for production
- Implement key rotation policies

### Risk Controls
- Position size limits
- Maximum drawdown stops
- Correlation limits
- Diversification requirements

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API keys and network connectivity
   # Verify rate limits and quotas
   ```

2. **Memory Usage**
   ```bash
   # Monitor system resources
   # Adjust data retention periods
   ```

3. **Signal Quality**
   ```bash
   # Verify data sources
   # Check confidence thresholds
   # Review backtesting results
   ```

### Debug Mode
```bash
python main.py analyze --debug
```

## 📞 Support

### Logs Location
- **Application**: `logs/crypto_analyzer.log`
- **Signals**: `logs/trading_signals.log`
- **Errors**: `logs/errors.log`
- **Performance**: `logs/performance.log`

### Monitoring Dashboard
- System health: http://localhost:8000/api/status
- Performance metrics: Available in web dashboard
- Alert history: Available in monitoring section

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider your risk tolerance before making investment decisions.

---

**Built with ❤️ by the Crypto Trend Analyzer team**