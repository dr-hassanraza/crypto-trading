import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from src.analyzers.technical_indicators import TechnicalAnalyzer
from src.signals.signal_generator import AdvancedSignalGenerator, SignalType
from config.config import Config

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    id: str
    symbol: str
    order_type: OrderType
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    fees: float
    status: OrderStatus
    signal_strength: float
    confidence_score: float

@dataclass
class BacktestResults:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade_duration: timedelta
    total_fees: float
    start_date: datetime
    end_date: datetime
    trades: List[Trade]
    equity_curve: pd.DataFrame
    performance_metrics: Dict[str, Any]

class BacktestEngine:
    def __init__(self):
        self.config = Config()
        self.technical_analyzer = TechnicalAnalyzer()
        self.signal_generator = AdvancedSignalGenerator()
        
        # Backtesting parameters
        self.initial_capital = 10000.0  # $10,000 starting capital
        self.position_size_pct = 0.1    # 10% position size
        self.transaction_fee = 0.001    # 0.1% transaction fee
        self.slippage = 0.0005         # 0.05% slippage
        
    def run_backtest(self, 
                     symbol: str, 
                     data: pd.DataFrame, 
                     start_date: datetime, 
                     end_date: datetime,
                     strategy_params: Optional[Dict[str, Any]] = None) -> BacktestResults:
        """Run comprehensive backtest on historical data."""
        
        logging.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Validate data
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
        
        # Initialize backtesting state
        portfolio = self._initialize_portfolio()
        trades = []
        open_positions = {}
        equity_curve = []
        
        # Calculate technical indicators
        data_with_indicators = self.technical_analyzer.calculate_all_indicators(data)
        
        # Main backtesting loop
        for idx, (timestamp, row) in enumerate(data_with_indicators.iterrows()):
            if idx < 50:  # Skip first 50 rows for indicator warmup
                continue
            
            current_price = row['close']
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, open_positions, current_price, symbol)
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions_value': portfolio_value - portfolio['cash']
            })
            
            # Check for exit signals on open positions
            if symbol in open_positions:
                exit_trade = self._check_exit_conditions(
                    open_positions[symbol], current_price, timestamp, row
                )
                if exit_trade:
                    completed_trade = self._close_position(
                        open_positions[symbol], current_price, timestamp, portfolio
                    )
                    trades.append(completed_trade)
                    del open_positions[symbol]
            
            # Generate trading signals
            if symbol not in open_positions:  # Only enter new positions if not already in one
                # Prepare market data for signal generation
                lookback_data = data_with_indicators.loc[:timestamp].tail(100)  # Last 100 periods
                
                if len(lookback_data) >= 50:
                    market_data = self._prepare_market_data_for_signals(symbol, lookback_data, row)
                    signal_data = self.signal_generator.generate_comprehensive_signals(market_data)
                    
                    # Check for entry signals
                    entry_trade = self._check_entry_conditions(
                        signal_data, current_price, timestamp, portfolio
                    )
                    if entry_trade:
                        open_positions[symbol] = entry_trade
                        self._update_portfolio_for_entry(portfolio, entry_trade, current_price)
        
        # Close any remaining open positions
        if open_positions:
            final_price = data_with_indicators['close'].iloc[-1]
            final_timestamp = data_with_indicators.index[-1]
            for position in open_positions.values():
                completed_trade = self._close_position(position, final_price, final_timestamp, portfolio)
                trades.append(completed_trade)
        
        # Calculate final results
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        results = self._calculate_backtest_results(
            trades, equity_df, start_date, end_date, symbol
        )
        
        logging.info(f"Backtest completed. Total return: {results.total_return:.2%}, "
                    f"Sharpe ratio: {results.sharpe_ratio:.2f}, Win rate: {results.win_rate:.2%}")
        
        return results
    
    def _initialize_portfolio(self) -> Dict[str, float]:
        """Initialize portfolio with starting capital."""
        return {
            'cash': self.initial_capital,
            'total_value': self.initial_capital
        }
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, float], 
                                  positions: Dict[str, Trade], 
                                  current_price: float, 
                                  symbol: str) -> float:
        """Calculate current portfolio value."""
        cash = portfolio['cash']
        positions_value = 0
        
        if symbol in positions:
            position = positions[symbol]
            positions_value = position.quantity * current_price
        
        return cash + positions_value
    
    def _prepare_market_data_for_signals(self, symbol: str, data: pd.DataFrame, current_row: pd.Series) -> Dict[str, Any]:
        """Prepare market data in format expected by signal generator."""
        
        # Simulate CoinGecko market data
        mock_coingecko_data = {
            'market_data': [{
                'current_price': current_row['close'],
                'market_cap': 1e9,  # Mock market cap
                'total_volume': current_row['volume'],
                'price_change_percentage_1h_in_currency': 0,  # Not available in backtest
                'price_change_percentage_24h': ((current_row['close'] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100) if len(data) > 1 else 0,
                'price_change_percentage_7d_in_currency': ((current_row['close'] - data['close'].iloc[-7]) / data['close'].iloc[-7] * 100) if len(data) > 7 else 0,
                'price_change_percentage_30d_in_currency': ((current_row['close'] - data['close'].iloc[-30]) / data['close'].iloc[-30] * 100) if len(data) > 30 else 0,
                'market_cap_rank': 10,
                'ath': data['high'].max(),
                'atl': data['low'].min(),
                'ath_change_percentage': (current_row['close'] - data['high'].max()) / data['high'].max() * 100,
                'atl_change_percentage': (current_row['close'] - data['low'].min()) / data['low'].min() * 100
            }]
        }
        
        return {
            'crypto_id': symbol.lower(),
            'symbol': symbol,
            'coingecko': mock_coingecko_data,
            'ohlcv_data': {
                '1d': data[['open', 'high', 'low', 'close', 'volume']],
                '4h': data[['open', 'high', 'low', 'close', 'volume']].resample('4H').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(),
                '1h': data[['open', 'high', 'low', 'close', 'volume']].resample('H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            },
            'news_sentiment': {
                'sentiment_score': 0,  # Neutral sentiment for backtest
                'article_count': 5,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'recent_headlines': []
            },
            'data_quality': {
                'score': 80,
                'grade': 'B'
            }
        }
    
    def _check_entry_conditions(self, signal_data: Dict[str, Any], 
                               current_price: float, 
                               timestamp: datetime, 
                               portfolio: Dict[str, float]) -> Optional[Trade]:
        """Check if entry conditions are met."""
        
        final_signal = signal_data.get('final_signal', {})
        signal = final_signal.get('signal', 'HOLD')
        confidence = final_signal.get('confidence_score', 0)
        strength = final_signal.get('strength', 0)
        
        # Entry thresholds
        min_confidence = 60
        min_strength = 0.3
        
        if signal in ['BUY', 'STRONG_BUY'] and confidence >= min_confidence and abs(strength) >= min_strength:
            # Calculate position size
            position_value = portfolio['cash'] * self.position_size_pct
            quantity = position_value / (current_price * (1 + self.slippage))  # Account for slippage
            
            # Check if we have enough cash
            total_cost = quantity * current_price * (1 + self.transaction_fee + self.slippage)
            if total_cost <= portfolio['cash']:
                return Trade(
                    id=f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{signal_data['crypto_id']}",
                    symbol=signal_data['symbol'],
                    order_type=OrderType.BUY,
                    entry_price=current_price * (1 + self.slippage),  # Account for slippage
                    exit_price=None,
                    quantity=quantity,
                    entry_time=timestamp,
                    exit_time=None,
                    stop_loss=final_signal.get('stop_loss'),
                    take_profit=final_signal.get('take_profit'),
                    pnl=None,
                    pnl_pct=None,
                    fees=quantity * current_price * self.transaction_fee,
                    status=OrderStatus.FILLED,
                    signal_strength=strength,
                    confidence_score=confidence
                )
        
        return None
    
    def _check_exit_conditions(self, position: Trade, 
                              current_price: float, 
                              timestamp: datetime, 
                              current_row: pd.Series) -> bool:
        """Check if exit conditions are met."""
        
        # Stop-loss check
        if position.stop_loss and current_price <= position.stop_loss:
            return True
        
        # Take-profit check  
        if position.take_profit and current_price >= position.take_profit:
            return True
        
        # Time-based exit (max holding period of 30 days)
        max_holding_days = 30
        if (timestamp - position.entry_time).days >= max_holding_days:
            return True
        
        # Technical exit conditions (simplified)
        # Exit if RSI becomes overbought (>80) for long positions
        if hasattr(current_row, 'RSI') and current_row['RSI'] > 80:
            return True
        
        return False
    
    def _close_position(self, position: Trade, 
                       exit_price: float, 
                       exit_time: datetime, 
                       portfolio: Dict[str, float]) -> Trade:
        """Close an open position."""
        
        # Account for slippage and fees
        actual_exit_price = exit_price * (1 - self.slippage)
        exit_fees = position.quantity * actual_exit_price * self.transaction_fee
        
        # Calculate P&L
        gross_pnl = (actual_exit_price - position.entry_price) * position.quantity
        net_pnl = gross_pnl - position.fees - exit_fees
        pnl_pct = net_pnl / (position.entry_price * position.quantity) * 100
        
        # Update portfolio
        portfolio['cash'] += (position.quantity * actual_exit_price) - exit_fees
        
        # Update trade record
        position.exit_price = actual_exit_price
        position.exit_time = exit_time
        position.pnl = net_pnl
        position.pnl_pct = pnl_pct
        position.fees += exit_fees
        
        return position
    
    def _update_portfolio_for_entry(self, portfolio: Dict[str, float], 
                                   trade: Trade, 
                                   current_price: float):
        """Update portfolio when entering a position."""
        total_cost = trade.quantity * trade.entry_price + trade.fees
        portfolio['cash'] -= total_cost
    
    def _calculate_backtest_results(self, trades: List[Trade], 
                                   equity_curve: pd.DataFrame, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   symbol: str) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        
        if not trades:
            # No trades case
            return BacktestResults(
                total_return=0.0,
                annualized_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_trade_duration=timedelta(0),
                total_fees=0.0,
                start_date=start_date,
                end_date=end_date,
                trades=trades,
                equity_curve=equity_curve,
                performance_metrics={}
            )
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
        
        # P&L statistics
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        winning_pnl = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        losing_pnl = sum(t.pnl for t in trades if t.pnl and t.pnl < 0)
        
        # Calculate metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        
        # Trade duration
        durations = [(t.exit_time - t.entry_time) for t in trades if t.exit_time]
        avg_trade_duration = sum(durations, timedelta(0)) / len(durations) if durations else timedelta(0)
        
        # Total fees
        total_fees = sum(t.fees for t in trades)
        
        # Portfolio performance
        initial_value = equity_curve['portfolio_value'].iloc[0]
        final_value = equity_curve['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Maximum drawdown
        equity_curve['peak'] = equity_curve['portfolio_value'].expanding().max()
        equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['peak']) / equity_curve['peak']
        max_drawdown = equity_curve['drawdown'].min()
        
        # Sharpe ratio
        daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 365) / (daily_returns.std() * np.sqrt(365))
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1 and negative_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() * 365) / (negative_returns.std() * np.sqrt(365))
        else:
            sortino_ratio = sharpe_ratio  # Fallback to Sharpe if no negative returns
        
        # Additional performance metrics
        performance_metrics = {
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'recovery_factor': total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'avg_monthly_return': (1 + total_return) ** (12 / (days / 30.44)) - 1 if days > 30 else 0,
            'volatility': daily_returns.std() * np.sqrt(365) if len(daily_returns) > 1 else 0,
            'best_trade': max(t.pnl for t in trades if t.pnl),
            'worst_trade': min(t.pnl for t in trades if t.pnl),
            'consecutive_wins': self._calculate_consecutive_wins(trades),
            'consecutive_losses': self._calculate_consecutive_losses(trades),
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss),
            'kelly_criterion': self._calculate_kelly_criterion(trades) if trades else 0
        }
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            total_fees=total_fees,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
            equity_curve=equity_curve,
            performance_metrics=performance_metrics
        )
    
    def _calculate_consecutive_wins(self, trades: List[Trade]) -> int:
        """Calculate maximum consecutive wins."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades: List[Trade]) -> int:
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_kelly_criterion(self, trades: List[Trade]) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        wins = [t.pnl_pct for t in trades if t.pnl_pct and t.pnl_pct > 0]
        losses = [abs(t.pnl_pct) for t in trades if t.pnl_pct and t.pnl_pct < 0]
        
        if not wins or not losses:
            return 0
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) / 100  # Convert percentage to decimal
        avg_loss = np.mean(losses) / 100  # Convert percentage to decimal
        
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_pct = (b * p - q) / b
        return max(0, min(kelly_pct, 0.25))  # Cap at 25% for risk management
    
    def compare_strategies(self, results_list: List[BacktestResults]) -> Dict[str, Any]:
        """Compare multiple backtest results."""
        if not results_list:
            return {}
        
        comparison = {
            'summary': {
                'total_strategies': len(results_list),
                'best_total_return': max(r.total_return for r in results_list),
                'best_sharpe_ratio': max(r.sharpe_ratio for r in results_list),
                'lowest_max_drawdown': max(r.max_drawdown for r in results_list),  # Closest to 0
                'highest_win_rate': max(r.win_rate for r in results_list)
            },
            'detailed_comparison': []
        }
        
        for i, result in enumerate(results_list):
            comparison['detailed_comparison'].append({
                'strategy_id': i,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor
            })
        
        # Rank strategies
        comparison['rankings'] = {
            'by_total_return': sorted(enumerate(results_list), 
                                    key=lambda x: x[1].total_return, reverse=True),
            'by_sharpe_ratio': sorted(enumerate(results_list), 
                                    key=lambda x: x[1].sharpe_ratio, reverse=True),
            'by_max_drawdown': sorted(enumerate(results_list), 
                                    key=lambda x: x[1].max_drawdown, reverse=True)  # Less negative is better
        }
        
        return comparison
    
    def generate_backtest_report(self, results: BacktestResults) -> str:
        """Generate a comprehensive backtest report."""
        report = f"""
=== CRYPTO TREND ANALYZER - BACKTEST REPORT ===

PERFORMANCE SUMMARY:
• Total Return: {results.total_return:.2%}
• Annualized Return: {results.annualized_return:.2%}
• Maximum Drawdown: {results.max_drawdown:.2%}
• Sharpe Ratio: {results.sharpe_ratio:.2f}
• Sortino Ratio: {results.sortino_ratio:.2f}

TRADE STATISTICS:
• Total Trades: {results.total_trades}
• Winning Trades: {results.winning_trades} ({results.win_rate:.1%})
• Losing Trades: {results.losing_trades} ({(1-results.win_rate):.1%})
• Profit Factor: {results.profit_factor:.2f}
• Average Win: ${results.avg_win:.2f}
• Average Loss: ${results.avg_loss:.2f}
• Average Trade Duration: {results.avg_trade_duration}

RISK METRICS:
• Best Trade: ${results.performance_metrics.get('best_trade', 0):.2f}
• Worst Trade: ${results.performance_metrics.get('worst_trade', 0):.2f}
• Max Consecutive Wins: {results.performance_metrics.get('consecutive_wins', 0)}
• Max Consecutive Losses: {results.performance_metrics.get('consecutive_losses', 0)}
• Kelly Criterion: {results.performance_metrics.get('kelly_criterion', 0):.1%}

PERIOD: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
TOTAL FEES PAID: ${results.total_fees:.2f}

EXPECTANCY: ${results.performance_metrics.get('expectancy', 0):.2f} per trade
"""
        return report