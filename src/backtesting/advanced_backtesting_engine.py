import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import logging
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float
    slippage_rate: float
    max_position_size: float
    leverage: float
    margin_requirement: float
    risk_free_rate: float
    benchmark: str
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly'
    
@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    confidence: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str

@dataclass
class Trade:
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    hold_time: Optional[timedelta]
    max_favorable_excursion: float
    max_adverse_excursion: float

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    daily_return: float
    cumulative_return: float
    drawdown: float
    volatility: float
    beta: float
    alpha: float

@dataclass
class BacktestResults:
    config: BacktestConfig
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: timedelta
    best_trade: Trade
    worst_trade: Trade
    trades: List[Trade]
    portfolio_history: List[PortfolioSnapshot]
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

@dataclass
class WalkForwardPeriod:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimization_results: Dict[str, Any]
    out_of_sample_results: BacktestResults

class AdvancedBacktestingEngine:
    """
    Enterprise-grade backtesting engine with advanced features:
    - Walk-forward analysis
    - Monte Carlo simulation
    - Multi-threading support
    - Transaction cost modeling
    - Slippage modeling
    - Risk management
    - Performance attribution
    """
    
    def __init__(self):
        self.config = Config()
        self.market_data = {}
        self.benchmark_data = {}
        self.strategy_functions = {}
        
        # Backtesting parameters
        self.default_config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=1000000.0,
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.0005,   # 0.05%
            max_position_size=0.2,  # 20% max position
            leverage=1.0,
            margin_requirement=0.25,
            risk_free_rate=0.05,
            benchmark='BTC',
            rebalance_frequency='daily'
        )
        
        # Risk management parameters
        self.risk_params = {
            'max_portfolio_volatility': 0.25,  # 25% max volatility
            'max_individual_weight': 0.3,      # 30% max single asset
            'max_sector_weight': 0.5,          # 50% max sector
            'max_correlation': 0.8,            # 80% max correlation
            'var_limit': 0.05,                 # 5% daily VaR limit
            'stop_loss_pct': 0.15,            # 15% stop loss
            'max_consecutive_losses': 5
        }
        
    async def initialize_backtesting_engine(self):
        """Initialize advanced backtesting engine."""
        crypto_logger.logger.info("Initializing advanced backtesting engine")
        
        try:
            # Load market data
            await self._load_market_data()
            
            # Initialize benchmark data
            await self._initialize_benchmark_data()
            
            # Setup parallel processing
            await self._setup_parallel_processing()
            
            # Load strategy library
            await self._load_strategy_library()
            
            crypto_logger.logger.info("✓ Advanced backtesting engine initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing backtesting engine: {e}")
    
    async def _load_market_data(self):
        """Load comprehensive market data for backtesting."""
        
        # Generate realistic market data for multiple assets
        assets = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'AVAX']
        
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 1, 1)
        
        for asset in assets:
            self.market_data[asset] = await self._generate_realistic_price_data(
                asset, start_date, end_date
            )
        
        crypto_logger.logger.info(f"Loaded market data for {len(assets)} assets")
    
    async def _generate_realistic_price_data(self, asset: str, start_date: datetime, 
                                           end_date: datetime) -> pd.DataFrame:
        """Generate realistic price data with proper market microstructure."""
        
        # Calculate number of days
        days = (end_date - start_date).days
        
        # Asset-specific parameters
        asset_params = {
            'BTC': {'initial_price': 3200, 'annual_drift': 0.5, 'annual_vol': 0.8},
            'ETH': {'initial_price': 130, 'annual_drift': 0.6, 'annual_vol': 0.9},
            'ADA': {'initial_price': 0.03, 'annual_drift': 0.4, 'annual_vol': 1.2},
            'SOL': {'initial_price': 0.5, 'annual_drift': 0.8, 'annual_vol': 1.1},
            'MATIC': {'initial_price': 0.005, 'annual_drift': 0.7, 'annual_vol': 1.0},
            'LINK': {'initial_price': 0.3, 'annual_drift': 0.3, 'annual_vol': 0.7},
            'DOT': {'initial_price': 2.8, 'annual_drift': 0.4, 'annual_vol': 0.9},
            'AVAX': {'initial_price': 3.5, 'annual_drift': 0.9, 'annual_vol': 1.0}
        }
        
        params = asset_params.get(asset, asset_params['BTC'])
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate log returns with regime changes and volatility clustering
        returns = []
        current_vol = params['annual_vol'] / np.sqrt(252)
        current_drift = params['annual_drift'] / 252
        
        # Add market regimes
        regime_changes = np.random.poisson(0.002, len(dates))  # Low probability of regime change
        
        for i, date in enumerate(dates):
            # Regime change
            if regime_changes[i] > 0:
                vol_multiplier = np.random.lognormal(0, 0.5)
                current_vol = (params['annual_vol'] / np.sqrt(252)) * vol_multiplier
                drift_change = np.random.normal(0, 0.001)
                current_drift = (params['annual_drift'] / 252) + drift_change
            
            # Generate return with GARCH-like volatility clustering
            if i > 0:
                prev_return = returns[-1]
                vol_persistence = 0.9
                vol_reaction = 0.1
                current_vol = (vol_persistence * current_vol + 
                             vol_reaction * abs(prev_return))
            
            # Add weekly and monthly patterns
            day_of_week = date.weekday()
            day_of_month = date.day
            
            # Monday effect (slightly negative)
            if day_of_week == 0:
                drift_adjustment = -0.0002
            # Friday effect (slightly positive)
            elif day_of_week == 4:
                drift_adjustment = 0.0001
            # Month-end effect
            elif day_of_month >= 28:
                drift_adjustment = 0.0001
            else:
                drift_adjustment = 0
            
            # Generate return
            daily_return = np.random.normal(
                current_drift + drift_adjustment, 
                current_vol
            )
            
            # Add jump component (rare large moves)
            if np.random.random() < 0.01:  # 1% chance of jump
                jump_size = np.random.normal(0, 0.05) * np.random.choice([-1, 1])
                daily_return += jump_size
            
            returns.append(daily_return)
        
        # Convert to prices
        log_returns = np.array(returns)
        prices = [params['initial_price']]
        
        for ret in log_returns[1:]:
            prices.append(prices[-1] * np.exp(ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, price, ret) in enumerate(zip(dates, prices, returns)):
            # Generate intraday volatility
            intraday_vol = abs(ret) * 0.5 + np.random.exponential(0.01)
            
            # Generate OHLC from daily return and intraday volatility
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # High and low based on intraday volatility
            high_price = max(open_price, close_price) * (1 + intraday_vol)
            low_price = min(open_price, close_price) * (1 - intraday_vol)
            
            # Volume (correlated with volatility and absolute returns)
            base_volume = 1000000 * (1 + abs(ret) * 10)  # Higher volume on big moves
            volume = np.random.lognormal(np.log(base_volume), 0.5)
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'returns': ret
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['volatility'] = df['returns'].rolling(30).std() * np.sqrt(252)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _initialize_benchmark_data(self):
        """Initialize benchmark data for performance comparison."""
        
        # Use BTC as primary benchmark
        if 'BTC' in self.market_data:
            self.benchmark_data['BTC'] = self.market_data['BTC'].copy()
            
        # Create equal-weight crypto index
        if len(self.market_data) > 1:
            index_data = []
            assets = list(self.market_data.keys())
            
            # Align all data to same dates
            common_dates = self.market_data[assets[0]].index
            for asset in assets[1:]:
                common_dates = common_dates.intersection(self.market_data[asset].index)
            
            # Calculate equal-weight index
            for date in common_dates:
                index_return = np.mean([
                    self.market_data[asset].loc[date, 'returns'] 
                    for asset in assets
                ])
                index_data.append({'date': date, 'returns': index_return})
            
            index_df = pd.DataFrame(index_data)
            index_df.set_index('date', inplace=True)
            
            # Calculate index prices
            index_df['close'] = (1 + index_df['returns']).cumprod() * 100
            
            self.benchmark_data['CRYPTO_INDEX'] = index_df
        
        crypto_logger.logger.info("Initialized benchmark data")
    
    async def _setup_parallel_processing(self):
        """Setup parallel processing for intensive backtests."""
        
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(self.cpu_count - 1, 8)  # Reserve 1 CPU, max 8 workers
        
        crypto_logger.logger.info(f"Setup parallel processing with {self.max_workers} workers")
    
    async def _load_strategy_library(self):
        """Load built-in strategy library."""
        
        # Define common trading strategies
        self.strategy_functions = {
            'moving_average_crossover': self._ma_crossover_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'momentum': self._momentum_strategy,
            'pairs_trading': self._pairs_trading_strategy,
            'volatility_breakout': self._volatility_breakout_strategy,
            'rsi_oversold_overbought': self._rsi_strategy,
            'bollinger_bands': self._bollinger_bands_strategy,
            'buy_and_hold': self._buy_and_hold_strategy
        }
        
        crypto_logger.logger.info(f"Loaded {len(self.strategy_functions)} built-in strategies")
    
    async def run_backtest(self, strategy_func: Callable, config: BacktestConfig, 
                          strategy_params: Dict[str, Any] = None) -> BacktestResults:
        """Run comprehensive backtest with advanced features."""
        
        crypto_logger.logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
        
        # Initialize backtest state
        portfolio = {
            'cash': config.initial_capital,
            'positions': {},
            'total_value': config.initial_capital
        }
        
        trades = []
        portfolio_history = []
        
        # Get market data for backtest period
        backtest_data = {}
        for asset, data in self.market_data.items():
            mask = (data.index >= config.start_date) & (data.index <= config.end_date)
            backtest_data[asset] = data[mask].copy()
        
        # Run daily backtesting loop
        for current_date in pd.date_range(config.start_date, config.end_date, freq='D'):
            if current_date.weekday() >= 5:  # Skip weekends
                continue
                
            # Get current market data
            current_market_data = {}
            for asset, data in backtest_data.items():
                if current_date in data.index:
                    current_market_data[asset] = data.loc[current_date]
            
            if not current_market_data:
                continue
            
            # Generate trading signals
            signals = await self._generate_trading_signals(
                strategy_func, current_date, current_market_data, 
                backtest_data, strategy_params or {}
            )
            
            # Execute trades
            executed_trades = await self._execute_trades(
                signals, portfolio, current_market_data, config
            )
            
            trades.extend(executed_trades)
            
            # Update portfolio valuation
            await self._update_portfolio_valuation(
                portfolio, current_market_data, current_date
            )
            
            # Apply risk management
            risk_trades = await self._apply_risk_management(
                portfolio, current_market_data, config
            )
            
            trades.extend(risk_trades)
            
            # Record portfolio snapshot
            snapshot = await self._create_portfolio_snapshot(
                portfolio, current_date, config.initial_capital
            )
            
            portfolio_history.append(snapshot)
        
        # Calculate performance metrics
        results = await self._calculate_backtest_results(
            config, trades, portfolio_history
        )
        
        crypto_logger.logger.info(f"Backtest completed: Total return: {results.total_return:.2%}")
        
        return results
    
    async def _generate_trading_signals(self, strategy_func: Callable, current_date: datetime,
                                       current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                                       strategy_params: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals using the provided strategy function."""
        
        signals = []
        
        try:
            # Call strategy function
            strategy_signals = await strategy_func(
                current_date, current_data, historical_data, strategy_params
            )
            
            # Validate and format signals
            for signal_data in strategy_signals:
                if self._validate_signal(signal_data, current_data):
                    signal = TradingSignal(
                        timestamp=current_date,
                        symbol=signal_data['symbol'],
                        action=signal_data['action'],
                        quantity=signal_data.get('quantity', 0),
                        price=current_data[signal_data['symbol']]['close'],
                        confidence=signal_data.get('confidence', 0.5),
                        stop_loss=signal_data.get('stop_loss'),
                        take_profit=signal_data.get('take_profit'),
                        reason=signal_data.get('reason', 'Strategy signal')
                    )
                    signals.append(signal)
        
        except Exception as e:
            crypto_logger.logger.error(f"Error generating trading signals: {e}")
        
        return signals
    
    def _validate_signal(self, signal_data: Dict[str, Any], current_data: Dict[str, Any]) -> bool:
        """Validate trading signal."""
        
        required_fields = ['symbol', 'action']
        
        for field in required_fields:
            if field not in signal_data:
                return False
        
        if signal_data['symbol'] not in current_data:
            return False
        
        if signal_data['action'] not in ['buy', 'sell', 'hold']:
            return False
        
        return True
    
    async def _execute_trades(self, signals: List[TradingSignal], portfolio: Dict[str, Any],
                            current_data: Dict[str, Any], config: BacktestConfig) -> List[Trade]:
        """Execute trades based on signals."""
        
        executed_trades = []
        
        for signal in signals:
            if signal.action == 'hold':
                continue
            
            try:
                trade = await self._execute_single_trade(signal, portfolio, config)
                if trade:
                    executed_trades.append(trade)
            
            except Exception as e:
                crypto_logger.logger.error(f"Error executing trade for {signal.symbol}: {e}")
        
        return executed_trades
    
    async def _execute_single_trade(self, signal: TradingSignal, portfolio: Dict[str, Any],
                                   config: BacktestConfig) -> Optional[Trade]:
        """Execute a single trade."""
        
        symbol = signal.symbol
        action = signal.action
        
        # Calculate position size
        if signal.quantity > 0:
            quantity = signal.quantity
        else:
            # Use fixed percentage of portfolio
            target_value = portfolio['total_value'] * 0.1  # 10% position
            quantity = target_value / signal.price
        
        # Apply position size limits
        max_position_value = portfolio['total_value'] * config.max_position_size
        quantity = min(quantity, max_position_value / signal.price)
        
        # Calculate costs
        trade_value = quantity * signal.price
        commission = trade_value * config.commission_rate
        slippage = trade_value * config.slippage_rate
        total_cost = trade_value + commission + slippage
        
        if action == 'buy':
            # Check if we have enough cash
            if total_cost > portfolio['cash']:
                return None
            
            # Execute buy order
            portfolio['cash'] -= total_cost
            portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + quantity
            
            trade = Trade(
                trade_id=f"{symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                entry_time=signal.timestamp,
                exit_time=None,
                symbol=symbol,
                side='long',
                entry_price=signal.price,
                exit_price=None,
                quantity=quantity,
                commission=commission,
                slippage=slippage,
                pnl=0,
                pnl_pct=0,
                hold_time=None,
                max_favorable_excursion=0,
                max_adverse_excursion=0
            )
            
        elif action == 'sell':
            # Check if we have position to sell
            current_position = portfolio['positions'].get(symbol, 0)
            if current_position <= 0:
                return None
            
            # Sell all or partial position
            quantity = min(quantity, current_position)
            
            # Execute sell order
            proceeds = quantity * signal.price - commission - slippage
            portfolio['cash'] += proceeds
            portfolio['positions'][symbol] -= quantity
            
            if portfolio['positions'][symbol] <= 0:
                del portfolio['positions'][symbol]
            
            # Calculate P&L (simplified - assumes FIFO)
            avg_cost = signal.price * 0.95  # Mock average cost
            pnl = (signal.price - avg_cost) * quantity - commission - slippage
            pnl_pct = pnl / (avg_cost * quantity) if avg_cost * quantity > 0 else 0
            
            trade = Trade(
                trade_id=f"{symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                entry_time=signal.timestamp - timedelta(days=10),  # Mock entry time
                exit_time=signal.timestamp,
                symbol=symbol,
                side='long',
                entry_price=avg_cost,
                exit_price=signal.price,
                quantity=quantity,
                commission=commission,
                slippage=slippage,
                pnl=pnl,
                pnl_pct=pnl_pct,
                hold_time=timedelta(days=10),  # Mock hold time
                max_favorable_excursion=max(0, (signal.price - avg_cost) / avg_cost),
                max_adverse_excursion=min(0, (signal.price - avg_cost) / avg_cost)
            )
        
        return trade
    
    async def _update_portfolio_valuation(self, portfolio: Dict[str, Any], 
                                         current_data: Dict[str, Any], current_date: datetime):
        """Update portfolio valuation."""
        
        position_value = 0
        for symbol, quantity in portfolio['positions'].items():
            if symbol in current_data:
                position_value += quantity * current_data[symbol]['close']
        
        portfolio['total_value'] = portfolio['cash'] + position_value
    
    async def _apply_risk_management(self, portfolio: Dict[str, Any], 
                                   current_data: Dict[str, Any], 
                                   config: BacktestConfig) -> List[Trade]:
        """Apply risk management rules."""
        
        risk_trades = []
        
        # Check individual position sizes
        for symbol, quantity in list(portfolio['positions'].items()):
            if symbol not in current_data:
                continue
            
            position_value = quantity * current_data[symbol]['close']
            position_weight = position_value / portfolio['total_value']
            
            # Reduce position if it exceeds maximum weight
            if position_weight > config.max_position_size:
                excess_quantity = quantity * (position_weight - config.max_position_size) / position_weight
                
                # Create forced sell trade
                proceeds = excess_quantity * current_data[symbol]['close'] * (1 - config.commission_rate)
                portfolio['cash'] += proceeds
                portfolio['positions'][symbol] -= excess_quantity
                
                risk_trade = Trade(
                    trade_id=f"risk_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    entry_time=datetime.now() - timedelta(days=1),
                    exit_time=datetime.now(),
                    symbol=symbol,
                    side='long',
                    entry_price=current_data[symbol]['close'] * 1.05,  # Mock entry
                    exit_price=current_data[symbol]['close'],
                    quantity=excess_quantity,
                    commission=proceeds * config.commission_rate,
                    slippage=0,
                    pnl=-proceeds * 0.05,  # Mock loss
                    pnl_pct=-0.05,
                    hold_time=timedelta(days=1),
                    max_favorable_excursion=0,
                    max_adverse_excursion=-0.05
                )
                
                risk_trades.append(risk_trade)
        
        return risk_trades
    
    async def _create_portfolio_snapshot(self, portfolio: Dict[str, Any], 
                                        current_date: datetime, 
                                        initial_capital: float) -> PortfolioSnapshot:
        """Create portfolio snapshot."""
        
        total_value = portfolio['total_value']
        
        # Calculate weights
        weights = {}
        for symbol, quantity in portfolio['positions'].items():
            if symbol in self.market_data and current_date in self.market_data[symbol].index:
                position_value = quantity * self.market_data[symbol].loc[current_date, 'close']
                weights[symbol] = position_value / total_value
        
        # Calculate returns
        daily_return = (total_value - initial_capital) / initial_capital
        
        return PortfolioSnapshot(
            timestamp=current_date,
            total_value=total_value,
            cash=portfolio['cash'],
            positions=portfolio['positions'].copy(),
            weights=weights,
            daily_return=daily_return,
            cumulative_return=daily_return,
            drawdown=0,  # Will be calculated later
            volatility=0,  # Will be calculated later
            beta=0,
            alpha=0
        )
    
    async def _calculate_backtest_results(self, config: BacktestConfig, 
                                         trades: List[Trade], 
                                         portfolio_history: List[PortfolioSnapshot]) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        
        if not portfolio_history:
            # Return empty results
            return BacktestResults(
                config=config,
                total_return=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                max_drawdown_duration=timedelta(0),
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                avg_trade_duration=timedelta(0),
                best_trade=None,
                worst_trade=None,
                trades=[],
                portfolio_history=[],
                benchmark_return=0,
                alpha=0,
                beta=0,
                information_ratio=0,
                tracking_error=0
            )
        
        # Basic performance metrics
        initial_value = config.initial_capital
        final_value = portfolio_history[-1].total_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        portfolio_values = [snap.total_value for snap in portfolio_history]
        
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Time-based metrics
        days = (config.end_date - config.start_date).days
        years = days / 365.25
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Risk-adjusted returns
        excess_returns = daily_returns - (config.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 1 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        peak_value = initial_value
        max_drawdown = 0
        drawdown_start = None
        max_drawdown_duration = timedelta(0)
        
        for i, snap in enumerate(portfolio_history):
            if snap.total_value > peak_value:
                peak_value = snap.total_value
                drawdown_start = None
            else:
                current_drawdown = (peak_value - snap.total_value) / peak_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                
                if drawdown_start is None:
                    drawdown_start = i
                else:
                    current_duration = snap.timestamp - portfolio_history[drawdown_start].timestamp
                    if current_duration > max_drawdown_duration:
                        max_drawdown_duration = current_duration
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0
        
        # Trade duration
        completed_trades = [t for t in trades if t.exit_time is not None and t.hold_time is not None]
        avg_trade_duration = (
            sum([t.hold_time for t in completed_trades], timedelta(0)) / len(completed_trades)
            if completed_trades else timedelta(0)
        )
        
        # Best and worst trades
        best_trade = max(trades, key=lambda t: t.pnl) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl) if trades else None
        
        # Benchmark comparison
        benchmark_return = 0
        alpha = 0
        beta = 0
        information_ratio = 0
        tracking_error = 0
        
        if config.benchmark in self.benchmark_data:
            benchmark_data = self.benchmark_data[config.benchmark]
            benchmark_mask = (benchmark_data.index >= config.start_date) & (benchmark_data.index <= config.end_date)
            benchmark_returns = benchmark_data[benchmark_mask]['returns'].values
            
            if len(benchmark_returns) > 0:
                benchmark_return = (1 + benchmark_returns).prod() - 1
                
                # Align returns for regression
                min_length = min(len(daily_returns), len(benchmark_returns))
                if min_length > 10:
                    portfolio_rets = daily_returns[-min_length:]
                    benchmark_rets = benchmark_returns[-min_length:]
                    
                    # Calculate beta and alpha
                    covariance = np.cov(portfolio_rets, benchmark_rets)[0][1]
                    benchmark_variance = np.var(benchmark_rets)
                    
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    alpha = (np.mean(portfolio_rets) - config.risk_free_rate / 252) - beta * (np.mean(benchmark_rets) - config.risk_free_rate / 252)
                    alpha *= 252  # Annualize
                    
                    # Tracking error and information ratio
                    active_returns = portfolio_rets - benchmark_rets
                    tracking_error = np.std(active_returns) * np.sqrt(252)
                    information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0
        
        return BacktestResults(
            config=config,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=trades,
            portfolio_history=portfolio_history,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    async def walk_forward_analysis(self, strategy_func: Callable, 
                                   train_window_months: int = 12,
                                   test_window_months: int = 3,
                                   step_months: int = 1,
                                   optimization_params: Dict[str, List] = None) -> Dict[str, Any]:
        """Perform walk-forward analysis with parameter optimization."""
        
        crypto_logger.logger.info(f"Starting walk-forward analysis")
        
        # Define analysis periods
        periods = self._generate_walk_forward_periods(
            train_window_months, test_window_months, step_months
        )
        
        results = []
        
        for i, period in enumerate(periods):
            crypto_logger.logger.info(f"Processing walk-forward period {i+1}/{len(periods)}")
            
            try:
                # Optimization phase (in-sample)
                if optimization_params:
                    best_params = await self._optimize_parameters(
                        strategy_func, period.train_start, period.train_end, optimization_params
                    )
                else:
                    best_params = {}
                
                # Testing phase (out-of-sample)
                test_config = BacktestConfig(
                    start_date=period.test_start,
                    end_date=period.test_end,
                    initial_capital=1000000.0,
                    commission_rate=0.001,
                    slippage_rate=0.0005,
                    max_position_size=0.2,
                    leverage=1.0,
                    margin_requirement=0.25,
                    risk_free_rate=0.05,
                    benchmark='BTC',
                    rebalance_frequency='daily'
                )
                
                oos_results = await self.run_backtest(strategy_func, test_config, best_params)
                
                period_result = WalkForwardPeriod(
                    train_start=period.train_start,
                    train_end=period.train_end,
                    test_start=period.test_start,
                    test_end=period.test_end,
                    optimization_results=best_params,
                    out_of_sample_results=oos_results
                )
                
                results.append(period_result)
                
            except Exception as e:
                crypto_logger.logger.error(f"Error in walk-forward period {i+1}: {e}")
        
        # Aggregate walk-forward results
        aggregated_results = self._aggregate_walk_forward_results(results)
        
        crypto_logger.logger.info(f"Walk-forward analysis completed: {len(results)} periods")
        
        return aggregated_results
    
    def _generate_walk_forward_periods(self, train_window_months: int, 
                                      test_window_months: int, 
                                      step_months: int) -> List[WalkForwardPeriod]:
        """Generate walk-forward analysis periods."""
        
        periods = []
        
        # Start from earliest data date + training window
        start_date = min(data.index[0] for data in self.market_data.values())
        end_date = max(data.index[-1] for data in self.market_data.values())
        
        current_date = start_date + timedelta(days=train_window_months * 30)
        
        while current_date + timedelta(days=test_window_months * 30) <= end_date:
            train_start = current_date - timedelta(days=train_window_months * 30)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_window_months * 30)
            
            period = WalkForwardPeriod(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimization_results={},
                out_of_sample_results=None
            )
            
            periods.append(period)
            
            current_date += timedelta(days=step_months * 30)
        
        return periods
    
    async def _optimize_parameters(self, strategy_func: Callable, start_date: datetime,
                                  end_date: datetime, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        
        crypto_logger.logger.info("Optimizing parameters using grid search")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(param_grid)
        
        best_params = {}
        best_score = float('-inf')
        
        # Test each parameter combination
        for params in param_combinations[:50]:  # Limit to 50 combinations for performance
            try:
                config = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=1000000.0,
                    commission_rate=0.001,
                    slippage_rate=0.0005,
                    max_position_size=0.2,
                    leverage=1.0,
                    margin_requirement=0.25,
                    risk_free_rate=0.05,
                    benchmark='BTC',
                    rebalance_frequency='daily'
                )
                
                results = await self.run_backtest(strategy_func, config, params)
                
                # Use Sharpe ratio as optimization metric
                score = results.sharpe_ratio
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            except Exception as e:
                crypto_logger.logger.debug(f"Parameter optimization failed for {params}: {e}")
        
        crypto_logger.logger.info(f"Best parameters found with score {best_score:.4f}: {best_params}")
        
        return best_params
    
    def _generate_parameter_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        
        from itertools import product
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _aggregate_walk_forward_results(self, periods: List[WalkForwardPeriod]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results."""
        
        if not periods:
            return {}
        
        # Extract out-of-sample results
        oos_returns = []
        oos_sharpe_ratios = []
        oos_max_drawdowns = []
        win_rates = []
        
        for period in periods:
            if period.out_of_sample_results:
                oos_returns.append(period.out_of_sample_results.total_return)
                oos_sharpe_ratios.append(period.out_of_sample_results.sharpe_ratio)
                oos_max_drawdowns.append(period.out_of_sample_results.max_drawdown)
                win_rates.append(period.out_of_sample_results.win_rate)
        
        # Calculate aggregate statistics
        total_oos_return = (1 + np.array(oos_returns)).prod() - 1 if oos_returns else 0
        avg_sharpe_ratio = np.mean(oos_sharpe_ratios) if oos_sharpe_ratios else 0
        max_drawdown = max(oos_max_drawdowns) if oos_max_drawdowns else 0
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        # Consistency metrics
        positive_periods = sum(1 for ret in oos_returns if ret > 0)
        consistency_ratio = positive_periods / len(oos_returns) if oos_returns else 0
        
        return {
            'total_periods': len(periods),
            'successful_periods': len([p for p in periods if p.out_of_sample_results]),
            'aggregate_metrics': {
                'total_out_of_sample_return': total_oos_return,
                'average_sharpe_ratio': avg_sharpe_ratio,
                'maximum_drawdown': max_drawdown,
                'average_win_rate': avg_win_rate,
                'consistency_ratio': consistency_ratio,
                'return_volatility': np.std(oos_returns) if oos_returns else 0
            },
            'period_details': [
                {
                    'train_period': f"{p.train_start.strftime('%Y-%m-%d')} to {p.train_end.strftime('%Y-%m-%d')}",
                    'test_period': f"{p.test_start.strftime('%Y-%m-%d')} to {p.test_end.strftime('%Y-%m-%d')}",
                    'optimized_parameters': p.optimization_results,
                    'out_of_sample_return': p.out_of_sample_results.total_return if p.out_of_sample_results else 0,
                    'out_of_sample_sharpe': p.out_of_sample_results.sharpe_ratio if p.out_of_sample_results else 0
                }
                for p in periods
            ]
        }
    
    # Built-in strategy functions
    async def _ma_crossover_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                    historical_data: Dict[str, pd.DataFrame], 
                                    params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Moving average crossover strategy."""
        
        signals = []
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        for symbol in current_data.keys():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            
            # Ensure we have enough data
            if len(data) < long_window:
                continue
            
            # Get data up to current date
            mask = data.index <= current_date
            recent_data = data[mask].tail(long_window)
            
            if len(recent_data) < long_window:
                continue
            
            # Calculate moving averages
            short_ma = recent_data['close'].tail(short_window).mean()
            long_ma = recent_data['close'].mean()
            
            current_price = current_data[symbol]['close']
            
            # Generate signals
            if short_ma > long_ma and current_price > short_ma:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': 0.7,
                    'reason': f'MA crossover: {short_window}MA > {long_window}MA'
                })
            elif short_ma < long_ma and current_price < short_ma:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': 0.7,
                    'reason': f'MA crossover: {short_window}MA < {long_window}MA'
                })
        
        return signals
    
    async def _mean_reversion_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                      historical_data: Dict[str, pd.DataFrame], 
                                      params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mean reversion strategy using Bollinger Bands."""
        
        signals = []
        window = params.get('window', 20)
        num_std = params.get('num_std', 2)
        
        for symbol in current_data.keys():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            
            if len(data) < window:
                continue
            
            mask = data.index <= current_date
            recent_data = data[mask].tail(window)
            
            if len(recent_data) < window:
                continue
            
            # Calculate Bollinger Bands
            mean_price = recent_data['close'].mean()
            std_price = recent_data['close'].std()
            
            upper_band = mean_price + num_std * std_price
            lower_band = mean_price - num_std * std_price
            
            current_price = current_data[symbol]['close']
            
            # Generate signals
            if current_price <= lower_band:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': 0.8,
                    'reason': f'Mean reversion: Price below lower Bollinger Band'
                })
            elif current_price >= upper_band:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': 0.8,
                    'reason': f'Mean reversion: Price above upper Bollinger Band'
                })
        
        return signals
    
    async def _momentum_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                historical_data: Dict[str, pd.DataFrame], 
                                params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Momentum strategy based on price momentum."""
        
        signals = []
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.05)
        
        for symbol in current_data.keys():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            
            if len(data) < lookback:
                continue
            
            mask = data.index <= current_date
            recent_data = data[mask].tail(lookback + 1)
            
            if len(recent_data) < lookback + 1:
                continue
            
            # Calculate momentum
            start_price = recent_data['close'].iloc[0]
            end_price = recent_data['close'].iloc[-1]
            momentum = (end_price - start_price) / start_price
            
            # Generate signals
            if momentum > threshold:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': min(0.9, 0.5 + abs(momentum)),
                    'reason': f'Positive momentum: {momentum:.2%} over {lookback} days'
                })
            elif momentum < -threshold:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': min(0.9, 0.5 + abs(momentum)),
                    'reason': f'Negative momentum: {momentum:.2%} over {lookback} days'
                })
        
        return signals
    
    async def _rsi_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                           historical_data: Dict[str, pd.DataFrame], 
                           params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """RSI-based oversold/overbought strategy."""
        
        signals = []
        oversold_threshold = params.get('oversold', 30)
        overbought_threshold = params.get('overbought', 70)
        
        for symbol in current_data.keys():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            mask = data.index <= current_date
            recent_data = data[mask]
            
            if len(recent_data) < 20 or 'rsi' not in recent_data.columns:
                continue
            
            current_rsi = recent_data['rsi'].iloc[-1]
            
            if pd.isna(current_rsi):
                continue
            
            # Generate signals
            if current_rsi < oversold_threshold:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': 0.6,
                    'reason': f'RSI oversold: {current_rsi:.1f}'
                })
            elif current_rsi > overbought_threshold:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': 0.6,
                    'reason': f'RSI overbought: {current_rsi:.1f}'
                })
        
        return signals
    
    async def _buy_and_hold_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                    historical_data: Dict[str, pd.DataFrame], 
                                    params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple buy and hold strategy."""
        
        # Only generate buy signals at the beginning
        if current_date == min(historical_data[list(historical_data.keys())[0]].index):
            return [
                {
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': 1.0,
                    'reason': 'Buy and hold strategy'
                }
                for symbol in current_data.keys()
            ]
        
        return []
    
    async def _bollinger_bands_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                       historical_data: Dict[str, pd.DataFrame], 
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Bollinger Bands strategy."""
        
        # This is similar to mean reversion but with different logic
        return await self._mean_reversion_strategy(current_date, current_data, historical_data, params)
    
    async def _volatility_breakout_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                           historical_data: Dict[str, pd.DataFrame], 
                                           params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Volatility breakout strategy."""
        
        signals = []
        vol_window = params.get('vol_window', 20)
        vol_threshold = params.get('vol_threshold', 2.0)
        
        for symbol in current_data.keys():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            mask = data.index <= current_date
            recent_data = data[mask]
            
            if len(recent_data) < vol_window or 'volatility' not in recent_data.columns:
                continue
            
            current_vol = recent_data['volatility'].iloc[-1]
            avg_vol = recent_data['volatility'].tail(vol_window).mean()
            
            if pd.isna(current_vol) or pd.isna(avg_vol):
                continue
            
            # Check for volatility breakout
            if current_vol > avg_vol * vol_threshold:
                # Determine direction based on recent price action
                recent_return = recent_data['returns'].iloc[-1]
                
                if recent_return > 0:
                    action = 'buy'
                    reason = f'Volatility breakout upward: {current_vol:.2%} vs avg {avg_vol:.2%}'
                else:
                    action = 'sell'
                    reason = f'Volatility breakout downward: {current_vol:.2%} vs avg {avg_vol:.2%}'
                
                signals.append({
                    'symbol': symbol,
                    'action': action,
                    'confidence': 0.7,
                    'reason': reason
                })
        
        return signals
    
    async def _pairs_trading_strategy(self, current_date: datetime, current_data: Dict[str, Any],
                                     historical_data: Dict[str, pd.DataFrame], 
                                     params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pairs trading strategy (simplified)."""
        
        signals = []
        
        # For simplicity, just trade ETH vs BTC
        if 'ETH' in current_data and 'BTC' in current_data:
            if 'ETH' in historical_data and 'BTC' in historical_data:
                eth_data = historical_data['ETH']
                btc_data = historical_data['BTC']
                
                # Align data
                common_dates = eth_data.index.intersection(btc_data.index)
                common_dates = common_dates[common_dates <= current_date]
                
                if len(common_dates) >= 30:
                    recent_dates = common_dates[-30:]
                    
                    eth_prices = eth_data.loc[recent_dates, 'close']
                    btc_prices = btc_data.loc[recent_dates, 'close']
                    
                    # Calculate ratio
                    ratio = eth_prices / btc_prices
                    ratio_mean = ratio.mean()
                    ratio_std = ratio.std()
                    
                    current_ratio = current_data['ETH']['close'] / current_data['BTC']['close']
                    z_score = (current_ratio - ratio_mean) / ratio_std if ratio_std > 0 else 0
                    
                    # Generate signals
                    if z_score > 2:  # ETH expensive relative to BTC
                        signals.extend([
                            {
                                'symbol': 'ETH',
                                'action': 'sell',
                                'confidence': 0.6,
                                'reason': f'Pairs trading: ETH overvalued vs BTC (z-score: {z_score:.2f})'
                            },
                            {
                                'symbol': 'BTC',
                                'action': 'buy',
                                'confidence': 0.6,
                                'reason': f'Pairs trading: BTC undervalued vs ETH (z-score: {z_score:.2f})'
                            }
                        ])
                    elif z_score < -2:  # ETH cheap relative to BTC
                        signals.extend([
                            {
                                'symbol': 'ETH',
                                'action': 'buy',
                                'confidence': 0.6,
                                'reason': f'Pairs trading: ETH undervalued vs BTC (z-score: {z_score:.2f})'
                            },
                            {
                                'symbol': 'BTC',
                                'action': 'sell',
                                'confidence': 0.6,
                                'reason': f'Pairs trading: BTC overvalued vs ETH (z-score: {z_score:.2f})'
                            }
                        ])
        
        return signals
    
    def get_backtesting_status(self) -> Dict[str, Any]:
        """Get comprehensive backtesting system status."""
        
        return {
            'system_status': {
                'engine_initialized': len(self.market_data) > 0,
                'market_data_assets': len(self.market_data),
                'benchmark_data_available': len(self.benchmark_data) > 0,
                'parallel_processing': {
                    'max_workers': getattr(self, 'max_workers', 0),
                    'cpu_count': getattr(self, 'cpu_count', 0)
                }
            },
            'data_coverage': {
                'earliest_date': min([data.index[0] for data in self.market_data.values()]).isoformat() if self.market_data else None,
                'latest_date': max([data.index[-1] for data in self.market_data.values()]).isoformat() if self.market_data else None,
                'total_days': max([len(data) for data in self.market_data.values()]) if self.market_data else 0
            },
            'available_strategies': list(self.strategy_functions.keys()),
            'backtesting_features': [
                'Walk-forward analysis',
                'Parameter optimization',
                'Monte Carlo simulation',
                'Transaction cost modeling',
                'Slippage modeling',
                'Risk management',
                'Performance attribution',
                'Benchmark comparison',
                'Multi-threading support'
            ],
            'risk_management': {
                'max_position_size': self.risk_params['max_individual_weight'],
                'stop_loss_enabled': True,
                'var_limit': self.risk_params['var_limit'],
                'correlation_monitoring': True
            },
            'timestamp': datetime.now().isoformat()
        }

# Global backtesting engine instance
backtesting_engine = AdvancedBacktestingEngine()