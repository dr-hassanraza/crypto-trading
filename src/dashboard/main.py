from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_sources.market_data import MarketDataFetcher
from src.data_sources.realtime_streams import RealTimeDataStreamer
from src.analyzers.ai_analysis import AIMarketAnalyzer
from src.signals.signal_generator import AdvancedSignalGenerator
from src.backtesting.backtest_engine import BacktestEngine
from config.config import Config

app = FastAPI(title="Crypto Trend Analyzer", version="1.0.0")
config = Config()

# Global instances
market_fetcher = None
realtime_streamer = None
ai_analyzer = AIMarketAnalyzer()
signal_generator = AdvancedSignalGenerator()
backtest_engine = BacktestEngine()

# WebSocket connections
active_connections: List[WebSocket] = []

# Data cache
data_cache = {
    'signals': {},
    'market_data': {},
    'portfolio': {},
    'performance': {}
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global market_fetcher, realtime_streamer
    
    print("Starting Crypto Trend Analyzer Dashboard...")
    
    # Initialize market data fetcher
    market_fetcher = MarketDataFetcher()
    
    # Initialize real-time streamer
    realtime_streamer = RealTimeDataStreamer()
    
    # Set up real-time callbacks
    realtime_streamer.add_callback('price_update', handle_price_update)
    realtime_streamer.add_callback('realtime_indicators', handle_indicator_update)
    realtime_streamer.add_callback('price_anomaly', handle_anomaly_alert)
    
    # Start background tasks
    asyncio.create_task(update_market_data())
    asyncio.create_task(generate_signals_periodically())
    
    print("Dashboard started successfully!")

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown."""
    if realtime_streamer:
        realtime_streamer.stop_streaming()
    print("Dashboard shutdown complete.")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Trend Analyzer</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </head>
    <body>
        <nav class="navbar navbar-dark bg-dark">
            <div class="container-fluid">
                <span class="navbar-brand mb-0 h1">🚀 Crypto Trend Analyzer AI Agent</span>
                <div class="d-flex">
                    <span class="badge bg-success me-2" id="status">Live</span>
                    <span class="text-light" id="last-update">Loading...</span>
                </div>
            </div>
        </nav>

        <div class="container-fluid mt-4">
            <!-- Overview Cards -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body">
                            <h5>Portfolio Value</h5>
                            <h3 id="portfolio-value">$0</h3>
                            <small id="portfolio-change">+0.00%</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body">
                            <h5>Active Signals</h5>
                            <h3 id="active-signals">0</h3>
                            <small id="signal-summary">BUY: 0 | SELL: 0</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body">
                            <h5>Market Sentiment</h5>
                            <h3 id="market-sentiment">NEUTRAL</h3>
                            <small id="fear-greed">F&G Index: 50</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white">
                        <div class="card-body">
                            <h5>Win Rate</h5>
                            <h3 id="win-rate">0%</h3>
                            <small id="total-trades">0 trades</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content Tabs -->
            <ul class="nav nav-tabs" role="tablist">
                <li class="nav-item">
                    <a class="nav-link active" data-bs-toggle="tab" href="#signals">Trading Signals</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" data-bs-toggle="tab" href="#charts">Price Charts</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" data-bs-toggle="tab" href="#portfolio">Portfolio</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" data-bs-toggle="tab" href="#backtest">Backtesting</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" data-bs-toggle="tab" href="#settings">Settings</a>
                </li>
            </ul>

            <div class="tab-content mt-3">
                <!-- Trading Signals Tab -->
                <div id="signals" class="tab-pane fade show active">
                    <div class="row">
                        <div class="col-md-8">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Active Trading Signals</h5>
                                </div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="signals-table">
                                            <thead>
                                                <tr>
                                                    <th>Symbol</th>
                                                    <th>Signal</th>
                                                    <th>Confidence</th>
                                                    <th>Price</th>
                                                    <th>Target</th>
                                                    <th>Stop Loss</th>
                                                    <th>Time</th>
                                                </tr>
                                            </thead>
                                            <tbody></tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5>AI Analysis</h5>
                                </div>
                                <div class="card-body" id="ai-analysis">
                                    <p class="text-muted">Loading AI analysis...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Price Charts Tab -->
                <div id="charts" class="tab-pane fade">
                    <div class="row">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-header d-flex justify-content-between">
                                    <h5>Price Charts</h5>
                                    <select class="form-select w-auto" id="symbol-select">
                                        <option value="BTCUSDT">Bitcoin (BTC)</option>
                                        <option value="ETHUSDT">Ethereum (ETH)</option>
                                        <option value="SOLUSDT">Solana (SOL)</option>
                                        <option value="ADAUSDT">Cardano (ADA)</option>
                                    </select>
                                </div>
                                <div class="card-body">
                                    <div id="price-chart" style="height: 500px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Technical Indicators</h5>
                                </div>
                                <div class="card-body">
                                    <div id="indicators-chart" style="height: 300px;"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Volume Analysis</h5>
                                </div>
                                <div class="card-body">
                                    <div id="volume-chart" style="height: 300px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Tab -->
                <div id="portfolio" class="tab-pane fade">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Portfolio Allocation</h5>
                                </div>
                                <div class="card-body">
                                    <div id="portfolio-chart" style="height: 300px;"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Performance</h5>
                                </div>
                                <div class="card-body">
                                    <div id="performance-chart" style="height: 300px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Backtesting Tab -->
                <div id="backtest" class="tab-pane fade">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Backtest Parameters</h5>
                                </div>
                                <div class="card-body">
                                    <form id="backtest-form">
                                        <div class="mb-3">
                                            <label>Symbol</label>
                                            <select class="form-control" id="bt-symbol">
                                                <option value="BTCUSDT">Bitcoin</option>
                                                <option value="ETHUSDT">Ethereum</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Start Date</label>
                                            <input type="date" class="form-control" id="bt-start-date">
                                        </div>
                                        <div class="mb-3">
                                            <label>End Date</label>
                                            <input type="date" class="form-control" id="bt-end-date">
                                        </div>
                                        <button type="submit" class="btn btn-primary">Run Backtest</button>
                                    </form>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-8">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Backtest Results</h5>
                                </div>
                                <div class="card-body" id="backtest-results">
                                    <p class="text-muted">Run a backtest to see results</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Settings Tab -->
                <div id="settings" class="tab-pane fade">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Trading Settings</h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label>Position Size (%)</label>
                                        <input type="range" class="form-range" min="1" max="20" value="10" id="position-size">
                                        <span id="position-size-value">10%</span>
                                    </div>
                                    <div class="mb-3">
                                        <label>Stop Loss (%)</label>
                                        <input type="range" class="form-range" min="1" max="20" value="5" id="stop-loss">
                                        <span id="stop-loss-value">5%</span>
                                    </div>
                                    <div class="mb-3">
                                        <label>Min Confidence (%)</label>
                                        <input type="range" class="form-range" min="30" max="90" value="60" id="min-confidence">
                                        <span id="min-confidence-value">60%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Notification Settings</h5>
                                </div>
                                <div class="card-body">
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="email-alerts" checked>
                                        <label class="form-check-label">Email Alerts</label>
                                    </div>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="desktop-alerts">
                                        <label class="form-check-label">Desktop Notifications</label>
                                    </div>
                                    <div class="mb-3">
                                        <label>Alert Frequency</label>
                                        <select class="form-control" id="alert-frequency">
                                            <option value="immediate">Immediate</option>
                                            <option value="hourly">Hourly</option>
                                            <option value="daily">Daily</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // WebSocket connection
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            };

            // Handle real-time updates
            function handleRealtimeUpdate(data) {
                if (data.type === 'price_update') {
                    updatePriceDisplay(data);
                } else if (data.type === 'signal_update') {
                    updateSignalsTable(data);
                } else if (data.type === 'portfolio_update') {
                    updatePortfolioDisplay(data);
                }
            }

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                loadInitialData();
                setupEventListeners();
                startPeriodicUpdates();
            });

            function loadInitialData() {
                fetch('/api/signals')
                    .then(response => response.json())
                    .then(data => updateSignalsTable(data));
                
                fetch('/api/portfolio')
                    .then(response => response.json())
                    .then(data => updatePortfolioDisplay(data));
                
                fetch('/api/market-overview')
                    .then(response => response.json())
                    .then(data => updateMarketOverview(data));
            }

            function setupEventListeners() {
                // Symbol selection for charts
                document.getElementById('symbol-select').addEventListener('change', function() {
                    loadPriceChart(this.value);
                });

                // Backtest form
                document.getElementById('backtest-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    runBacktest();
                });

                // Settings sliders
                ['position-size', 'stop-loss', 'min-confidence'].forEach(id => {
                    const slider = document.getElementById(id);
                    const valueSpan = document.getElementById(id + '-value');
                    slider.addEventListener('input', function() {
                        valueSpan.textContent = this.value + '%';
                    });
                });
            }

            function startPeriodicUpdates() {
                // Update every 30 seconds
                setInterval(loadInitialData, 30000);
                
                // Update last update time
                setInterval(function() {
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                }, 1000);
            }

            function updateSignalsTable(signals) {
                const tbody = document.querySelector('#signals-table tbody');
                tbody.innerHTML = '';
                
                if (signals && signals.length > 0) {
                    signals.forEach(signal => {
                        const row = tbody.insertRow();
                        const badgeClass = signal.signal === 'BUY' ? 'bg-success' : 
                                          signal.signal === 'SELL' ? 'bg-danger' : 'bg-secondary';
                        
                        row.innerHTML = `
                            <td>${signal.symbol}</td>
                            <td><span class="badge ${badgeClass}">${signal.signal}</span></td>
                            <td>${signal.confidence}%</td>
                            <td>$${signal.price}</td>
                            <td>$${signal.target || 'N/A'}</td>
                            <td>$${signal.stop_loss || 'N/A'}</td>
                            <td>${new Date(signal.timestamp).toLocaleTimeString()}</td>
                        `;
                    });
                }
                
                // Update summary cards
                const buySignals = signals.filter(s => s.signal === 'BUY').length;
                const sellSignals = signals.filter(s => s.signal === 'SELL').length;
                document.getElementById('active-signals').textContent = signals.length;
                document.getElementById('signal-summary').textContent = `BUY: ${buySignals} | SELL: ${sellSignals}`;
            }

            function updatePortfolioDisplay(portfolio) {
                if (portfolio.total_value) {
                    document.getElementById('portfolio-value').textContent = 
                        '$' + portfolio.total_value.toLocaleString();
                    document.getElementById('portfolio-change').textContent = 
                        (portfolio.daily_change >= 0 ? '+' : '') + portfolio.daily_change.toFixed(2) + '%';
                }
            }

            function updateMarketOverview(overview) {
                if (overview.market_sentiment) {
                    document.getElementById('market-sentiment').textContent = overview.market_sentiment;
                }
                if (overview.fear_greed_index) {
                    document.getElementById('fear-greed').textContent = 
                        `F&G Index: ${overview.fear_greed_index.value}`;
                }
            }

            function loadPriceChart(symbol) {
                fetch(`/api/chart-data/${symbol}`)
                    .then(response => response.json())
                    .then(data => {
                        const trace = {
                            x: data.timestamps,
                            open: data.open,
                            high: data.high,
                            low: data.low,
                            close: data.close,
                            type: 'candlestick',
                            name: symbol
                        };
                        
                        const layout = {
                            title: `${symbol} Price Chart`,
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Price (USD)' },
                            height: 500
                        };
                        
                        Plotly.newPlot('price-chart', [trace], layout);
                    });
            }

            function runBacktest() {
                const symbol = document.getElementById('bt-symbol').value;
                const startDate = document.getElementById('bt-start-date').value;
                const endDate = document.getElementById('bt-end-date').value;
                
                document.getElementById('backtest-results').innerHTML = '<p>Running backtest...</p>';
                
                fetch('/api/backtest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symbol: symbol,
                        start_date: startDate,
                        end_date: endDate
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('backtest-results').innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Performance</h6>
                                <p>Total Return: <strong>${(data.total_return * 100).toFixed(2)}%</strong></p>
                                <p>Sharpe Ratio: <strong>${data.sharpe_ratio.toFixed(2)}</strong></p>
                                <p>Max Drawdown: <strong>${(data.max_drawdown * 100).toFixed(2)}%</strong></p>
                            </div>
                            <div class="col-md-6">
                                <h6>Trading Stats</h6>
                                <p>Total Trades: <strong>${data.total_trades}</strong></p>
                                <p>Win Rate: <strong>${(data.win_rate * 100).toFixed(1)}%</strong></p>
                                <p>Profit Factor: <strong>${data.profit_factor.toFixed(2)}</strong></p>
                            </div>
                        </div>
                    `;
                })
                .catch(error => {
                    document.getElementById('backtest-results').innerHTML = 
                        '<p class="text-danger">Error running backtest: ' + error.message + '</p>';
                });
            }

            // Load initial chart
            setTimeout(() => loadPriceChart('BTCUSDT'), 1000);
        </script>
    </body>
    </html>
    """

@app.get("/api/signals")
async def get_active_signals():
    """Get current active trading signals."""
    try:
        # Return cached signals or generate new ones
        if 'signals' in data_cache and data_cache['signals']:
            return list(data_cache['signals'].values())
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio status."""
    try:
        # Mock portfolio data for demo
        return {
            'total_value': 12500.50,
            'daily_change': 2.34,
            'positions': [
                {'symbol': 'BTC', 'value': 5000, 'pnl': 234.5},
                {'symbol': 'ETH', 'value': 3000, 'pnl': 123.4},
                {'symbol': 'SOL', 'value': 2000, 'pnl': -45.2}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-overview")
async def get_market_overview():
    """Get market overview and sentiment."""
    try:
        if market_fetcher:
            async with market_fetcher as fetcher:
                overview = await fetcher.get_market_overview()
                return overview
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart-data/{symbol}")
async def get_chart_data(symbol: str):
    """Get price chart data for a symbol."""
    try:
        # Mock chart data - in real implementation, fetch from data source
        timestamps = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        np.random.seed(42)
        
        prices = []
        base_price = 45000 if symbol == 'BTCUSDT' else 2500
        
        for i in range(len(timestamps)):
            price = base_price + np.random.normal(0, base_price * 0.02)
            prices.append(price)
            base_price = price
        
        return {
            'timestamps': [ts.isoformat() for ts in timestamps],
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest_api(backtest_params: dict):
    """Run backtest with given parameters."""
    try:
        symbol = backtest_params.get('symbol', 'BTCUSDT')
        start_date = datetime.fromisoformat(backtest_params.get('start_date'))
        end_date = datetime.fromisoformat(backtest_params.get('end_date'))
        
        # Generate mock historical data for backtesting
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        base_price = 45000 if symbol == 'BTCUSDT' else 2500
        prices = []
        volumes = []
        
        for i in range(len(dates)):
            price = base_price + np.random.normal(0, base_price * 0.02)
            volume = np.random.uniform(1000, 10000)
            prices.append(price)
            volumes.append(volume)
            base_price = price * (1 + np.random.normal(0, 0.001))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * np.random.uniform(1.001, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 0.999) for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Run backtest
        results = backtest_engine.run_backtest(symbol, data, start_date, end_date)
        
        return {
            'total_return': results.total_return,
            'annualized_return': results.annualized_return,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades,
            'winning_trades': results.winning_trades,
            'losing_trades': results.losing_trades
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status and health metrics."""
    return {
        'status': 'running',
        'realtime_streaming': realtime_streamer.is_streaming_active() if realtime_streamer else False,
        'last_update': datetime.now().isoformat(),
        'active_connections': len(active_connections),
        'cached_signals': len(data_cache.get('signals', {}))
    }

# Background Tasks
async def update_market_data():
    """Background task to update market data periodically."""
    while True:
        try:
            if market_fetcher:
                async with market_fetcher as fetcher:
                    # Update data for configured cryptos
                    for i, crypto_id in enumerate(config.CRYPTOS[:4]):  # Limit to 4 for demo
                        symbol = config.SYMBOLS[i] if i < len(config.SYMBOLS) else 'BTCUSDT'
                        comprehensive_data = await fetcher.get_comprehensive_data(crypto_id, symbol)
                        
                        if comprehensive_data:
                            data_cache['market_data'][crypto_id] = comprehensive_data
                        
                        # Add small delay between requests
                        await asyncio.sleep(1)
            
            await asyncio.sleep(300)  # Update every 5 minutes
            
        except Exception as e:
            print(f"Error updating market data: {e}")
            await asyncio.sleep(60)

async def generate_signals_periodically():
    """Background task to generate trading signals periodically."""
    while True:
        try:
            # Generate signals for cached market data
            for crypto_id, market_data in data_cache.get('market_data', {}).items():
                signal_data = signal_generator.generate_comprehensive_signals(market_data)
                
                if signal_data:
                    data_cache['signals'][crypto_id] = {
                        'symbol': signal_data['symbol'],
                        'signal': signal_data['final_signal']['signal'],
                        'confidence': signal_data['final_signal']['confidence_score'],
                        'price': signal_data['final_signal']['entry_price'],
                        'target': signal_data['final_signal']['take_profit'],
                        'stop_loss': signal_data['final_signal']['stop_loss'],
                        'timestamp': signal_data['timestamp'].isoformat()
                    }
                    
                    # Send real-time update to connected clients
                    await broadcast_update({
                        'type': 'signal_update',
                        'data': data_cache['signals'][crypto_id]
                    })
            
            await asyncio.sleep(180)  # Generate signals every 3 minutes
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            await asyncio.sleep(60)

async def broadcast_update(message: dict):
    """Broadcast update to all connected WebSocket clients."""
    if active_connections:
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

# Real-time data handlers
def handle_price_update(data):
    """Handle price updates from real-time streamer."""
    asyncio.create_task(broadcast_update({
        'type': 'price_update',
        'data': data
    }))

def handle_indicator_update(data):
    """Handle technical indicator updates."""
    asyncio.create_task(broadcast_update({
        'type': 'indicator_update',
        'data': data
    }))

def handle_anomaly_alert(data):
    """Handle price anomaly alerts."""
    asyncio.create_task(broadcast_update({
        'type': 'anomaly_alert',
        'data': data
    }))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")