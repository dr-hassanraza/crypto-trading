import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from config.config import Config

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

class RiskLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    VERY_AGGRESSIVE = "VERY_AGGRESSIVE"

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    risk_score: float = 0.0
    position_size_usd: float = 0.0
    allocation_pct: float = 0.0

@dataclass
class PortfolioMetrics:
    total_value: float
    cash_balance: float
    invested_value: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    portfolio_risk_score: float
    diversification_score: float
    leverage_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    positions_count: int
    largest_position_pct: float
    exposure_by_sector: Dict[str, float] = field(default_factory=dict)

class PortfolioManager:
    def __init__(self, initial_capital: float = 10000.0):
        self.config = Config()
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.risk_limits = self._set_default_risk_limits()
        self.performance_tracking = {
            'daily_returns': [],
            'portfolio_values': [],
            'timestamps': []
        }
    
    def _set_default_risk_limits(self) -> Dict[str, float]:
        """Set default risk management limits."""
        return {
            'max_position_size': 0.15,      # 15% max per position
            'max_portfolio_risk': 0.02,     # 2% max portfolio risk per day
            'max_sector_exposure': 0.40,    # 40% max exposure to any sector
            'max_correlation_exposure': 0.30, # 30% max in highly correlated assets
            'min_cash_reserve': 0.10,       # 10% minimum cash reserve
            'max_leverage': 1.0,            # No leverage by default
            'var_limit': 0.05,              # 5% VaR limit
            'max_drawdown_limit': 0.20      # 20% max drawdown limit
        }
    
    def add_position(self, symbol: str, quantity: float, entry_price: float, 
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """Add a new position to the portfolio."""
        
        position_value = quantity * entry_price
        
        # Pre-trade risk checks
        if not self._validate_new_position(symbol, position_value):
            logging.warning(f"Position rejected for {symbol} due to risk limits")
            return False
        
        # Check available cash
        if position_value > self.cash_balance:
            logging.warning(f"Insufficient cash for {symbol} position")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_usd=position_value,
            allocation_pct=position_value / self.get_total_portfolio_value()
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash_balance -= position_value
        
        # Calculate position risk score
        position.risk_score = self._calculate_position_risk(position)
        
        logging.info(f"Added position: {symbol} - {quantity} @ ${entry_price}")
        self._record_portfolio_snapshot()
        
        return True
    
    def close_position(self, symbol: str, exit_price: float, partial_quantity: Optional[float] = None) -> bool:
        """Close a position (fully or partially)."""
        
        if symbol not in self.positions:
            logging.warning(f"No open position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        close_quantity = partial_quantity or position.quantity
        
        if close_quantity > position.quantity:
            logging.warning(f"Cannot close more than available quantity for {symbol}")
            return False
        
        # Calculate P&L
        pnl = (exit_price - position.entry_price) * close_quantity
        pnl_pct = pnl / (position.entry_price * close_quantity) * 100
        
        # Update cash balance
        self.cash_balance += close_quantity * exit_price
        
        # Handle partial vs full close
        if close_quantity == position.quantity:
            # Full close
            position.status = PositionStatus.CLOSED
            position.unrealized_pnl = pnl
            position.unrealized_pnl_pct = pnl_pct
            
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            logging.info(f"Closed full position: {symbol} - P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            # Partial close
            position.quantity -= close_quantity
            position.position_size_usd = position.quantity * position.entry_price
            
            # Create record of partial close
            partial_position = Position(
                symbol=symbol + "_PARTIAL",
                quantity=close_quantity,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                status=PositionStatus.CLOSED,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct
            )
            self.closed_positions.append(partial_position)
            
            logging.info(f"Partially closed position: {symbol} - {close_quantity} shares, P&L: ${pnl:.2f}")
        
        self._record_portfolio_snapshot()
        return True
    
    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                old_price = position.current_price or position.entry_price
                position.current_price = price_data[symbol]
                position.position_size_usd = position.quantity * position.current_price
                
                # Update unrealized P&L
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                position.unrealized_pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
                
                # Check stop loss and take profit
                self._check_exit_conditions(position)
        
        # Update portfolio allocation percentages
        total_value = self.get_total_portfolio_value()
        for position in self.positions.values():
            position.allocation_pct = position.position_size_usd / total_value
        
        self._record_portfolio_snapshot()
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        
        total_value = self.get_total_portfolio_value()
        invested_value = sum(pos.position_size_usd for pos in self.positions.values())
        
        # P&L calculations
        total_unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in self.positions.values())
        total_realized_pnl = sum(pos.unrealized_pnl or 0 for pos in self.closed_positions)
        total_pnl = total_unrealized_pnl + total_realized_pnl
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Daily P&L
        daily_pnl, daily_pnl_pct = self._calculate_daily_pnl()
        
        # Risk metrics
        portfolio_risk_score = self._calculate_portfolio_risk()
        diversification_score = self._calculate_diversification_score()
        var_95 = self._calculate_var_95()
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Position metrics
        positions_count = len(self.positions)
        largest_position_pct = max([pos.allocation_pct for pos in self.positions.values()]) if self.positions else 0
        
        # Sector exposure
        exposure_by_sector = self._calculate_sector_exposure()
        
        return PortfolioMetrics(
            total_value=total_value,
            cash_balance=self.cash_balance,
            invested_value=invested_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            portfolio_risk_score=portfolio_risk_score,
            diversification_score=diversification_score,
            leverage_ratio=invested_value / total_value if total_value > 0 else 0,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            positions_count=positions_count,
            largest_position_pct=largest_position_pct,
            exposure_by_sector=exposure_by_sector
        )
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment."""
        metrics = self.get_portfolio_metrics()
        
        risk_flags = []
        risk_score = 0  # 0-100 scale
        
        # Check position size limits
        if metrics.largest_position_pct > self.risk_limits['max_position_size']:
            risk_flags.append(f"Large position concentration: {metrics.largest_position_pct:.1%}")
            risk_score += 20
        
        # Check cash reserves
        cash_pct = metrics.cash_balance / metrics.total_value
        if cash_pct < self.risk_limits['min_cash_reserve']:
            risk_flags.append(f"Low cash reserves: {cash_pct:.1%}")
            risk_score += 15
        
        # Check VaR
        if metrics.var_95 > self.risk_limits['var_limit']:
            risk_flags.append(f"High VaR: {metrics.var_95:.1%}")
            risk_score += 25
        
        # Check max drawdown
        if abs(metrics.max_drawdown) > self.risk_limits['max_drawdown_limit']:
            risk_flags.append(f"High drawdown: {metrics.max_drawdown:.1%}")
            risk_score += 30
        
        # Check diversification
        if metrics.diversification_score < 0.5:
            risk_flags.append("Poor diversification")
            risk_score += 20
        
        # Check sector concentration
        max_sector_exposure = max(metrics.exposure_by_sector.values()) if metrics.exposure_by_sector else 0
        if max_sector_exposure > self.risk_limits['max_sector_exposure']:
            risk_flags.append(f"High sector concentration: {max_sector_exposure:.1%}")
            risk_score += 15
        
        # Risk level classification
        if risk_score <= 20:
            risk_level = RiskLevel.CONSERVATIVE
        elif risk_score <= 40:
            risk_level = RiskLevel.MODERATE
        elif risk_score <= 70:
            risk_level = RiskLevel.AGGRESSIVE
        else:
            risk_level = RiskLevel.VERY_AGGRESSIVE
        
        return {
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level.value,
            'risk_flags': risk_flags,
            'portfolio_risk_score': metrics.portfolio_risk_score,
            'var_95': metrics.var_95,
            'max_drawdown': metrics.max_drawdown,
            'diversification_score': metrics.diversification_score,
            'leverage_ratio': metrics.leverage_ratio,
            'recommendations': self._generate_risk_recommendations(risk_flags, metrics)
        }
    
    def optimize_portfolio(self) -> Dict[str, Any]:
        """Suggest portfolio optimizations."""
        metrics = self.get_portfolio_metrics()
        risk_assessment = self.get_risk_assessment()
        
        recommendations = []
        
        # Position sizing recommendations
        if metrics.largest_position_pct > 0.20:
            recommendations.append({
                'type': 'position_sizing',
                'action': 'REDUCE',
                'description': f"Consider reducing largest position from {metrics.largest_position_pct:.1%} to <20%",
                'priority': 'HIGH'
            })
        
        # Diversification recommendations
        if metrics.diversification_score < 0.6:
            recommendations.append({
                'type': 'diversification',
                'action': 'ADD_POSITIONS',
                'description': "Consider adding more diverse positions to improve risk-adjusted returns",
                'priority': 'MEDIUM'
            })
        
        # Cash management
        cash_pct = metrics.cash_balance / metrics.total_value
        if cash_pct > 0.30:
            recommendations.append({
                'type': 'cash_management',
                'action': 'DEPLOY_CASH',
                'description': f"High cash allocation ({cash_pct:.1%}) - consider investing some cash",
                'priority': 'LOW'
            })
        elif cash_pct < 0.05:
            recommendations.append({
                'type': 'cash_management',
                'action': 'RAISE_CASH',
                'description': f"Low cash reserves ({cash_pct:.1%}) - consider taking some profits",
                'priority': 'HIGH'
            })
        
        # Risk management
        if risk_assessment['risk_score'] > 60:
            recommendations.append({
                'type': 'risk_management',
                'action': 'REDUCE_RISK',
                'description': "Portfolio risk is elevated - consider reducing position sizes or adding hedges",
                'priority': 'HIGH'
            })
        
        return {
            'recommendations': recommendations,
            'current_allocation': {pos.symbol: pos.allocation_pct for pos in self.positions.values()},
            'optimal_allocation': self._calculate_optimal_allocation(),
            'rebalancing_needed': len(recommendations) > 0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.get_portfolio_metrics()
        
        # Trade statistics
        all_trades = self.closed_positions
        winning_trades = [t for t in all_trades if (t.unrealized_pnl or 0) > 0]
        losing_trades = [t for t in all_trades if (t.unrealized_pnl or 0) < 0]
        
        win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
        avg_win = np.mean([t.unrealized_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.unrealized_pnl for t in losing_trades]) if losing_trades else 0
        
        # Time-based performance
        performance_periods = self._calculate_period_returns()
        
        return {
            'overview': {
                'total_return': metrics.total_pnl_pct,
                'total_value': metrics.total_value,
                'initial_capital': self.initial_capital,
                'cash_balance': metrics.cash_balance,
                'invested_value': metrics.invested_value
            },
            'risk_metrics': {
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'var_95': metrics.var_95,
                'portfolio_risk_score': metrics.portfolio_risk_score
            },
            'trade_statistics': {
                'total_trades': len(all_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            },
            'period_returns': performance_periods,
            'current_positions': {
                pos.symbol: {
                    'allocation': pos.allocation_pct,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                } for pos in self.positions.values()
            }
        }
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.position_size_usd for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def _validate_new_position(self, symbol: str, position_value: float) -> bool:
        """Validate new position against risk limits."""
        
        total_value = self.get_total_portfolio_value()
        position_pct = position_value / total_value
        
        # Check position size limit
        if position_pct > self.risk_limits['max_position_size']:
            return False
        
        # Check cash reserve limit
        remaining_cash = self.cash_balance - position_value
        if remaining_cash / total_value < self.risk_limits['min_cash_reserve']:
            return False
        
        # Check if adding this position would exceed sector limits
        sector_exposure = self._calculate_sector_exposure()
        symbol_sector = self._get_symbol_sector(symbol)
        current_sector_exposure = sector_exposure.get(symbol_sector, 0)
        
        if current_sector_exposure + position_pct > self.risk_limits['max_sector_exposure']:
            return False
        
        return True
    
    def _calculate_position_risk(self, position: Position) -> float:
        """Calculate risk score for a position (0-100)."""
        risk_score = 0
        
        # Size risk
        if position.allocation_pct > 0.15:
            risk_score += 30
        elif position.allocation_pct > 0.10:
            risk_score += 20
        elif position.allocation_pct > 0.05:
            risk_score += 10
        
        # Stop loss risk
        if not position.stop_loss:
            risk_score += 20
        else:
            stop_loss_distance = abs(position.entry_price - position.stop_loss) / position.entry_price
            if stop_loss_distance > 0.10:  # >10% stop loss
                risk_score += 15
            elif stop_loss_distance > 0.05:  # >5% stop loss
                risk_score += 10
        
        # Volatility risk (simplified - would use historical volatility in practice)
        symbol_volatility = self._get_symbol_volatility(position.symbol)
        if symbol_volatility > 0.05:  # >5% daily volatility
            risk_score += 25
        elif symbol_volatility > 0.03:  # >3% daily volatility
            risk_score += 15
        
        return min(risk_score, 100)
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate overall portfolio risk score."""
        if not self.positions:
            return 0
        
        # Weighted average of position risks
        total_value = sum(pos.position_size_usd for pos in self.positions.values())
        weighted_risk = sum(
            (pos.position_size_usd / total_value) * pos.risk_score 
            for pos in self.positions.values()
        )
        
        # Adjust for concentration risk
        concentration_penalty = self._calculate_concentration_risk()
        
        return min(weighted_risk + concentration_penalty, 100)
    
    def _calculate_diversification_score(self) -> float:
        """Calculate diversification score (0-1, higher is better)."""
        if len(self.positions) <= 1:
            return 0.0
        
        # Herfindahl-Hirschman Index for concentration
        allocations = [pos.allocation_pct for pos in self.positions.values()]
        hhi = sum(alloc ** 2 for alloc in allocations)
        
        # Convert HHI to diversification score (inverted and normalized)
        max_hhi = 1.0  # Maximum concentration (one position)
        min_hhi = 1.0 / len(self.positions)  # Perfect diversification
        
        if max_hhi == min_hhi:
            return 1.0
        
        diversification = 1 - ((hhi - min_hhi) / (max_hhi - min_hhi))
        return max(0, min(1, diversification))
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk."""
        if len(self.performance_tracking['daily_returns']) < 30:
            return 0.0
        
        daily_returns = np.array(self.performance_tracking['daily_returns'][-252:])  # Last year
        return np.percentile(daily_returns, 5) * -1  # 95% VaR (positive value)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history."""
        if len(self.performance_tracking['portfolio_values']) < 2:
            return 0.0
        
        values = np.array(self.performance_tracking['portfolio_values'])
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        return np.min(drawdown)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.performance_tracking['daily_returns']) < 30:
            return 0.0
        
        daily_returns = np.array(self.performance_tracking['daily_returns'])
        excess_returns = daily_returns  # Assuming risk-free rate = 0
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_daily_pnl(self) -> Tuple[float, float]:
        """Calculate daily P&L."""
        if len(self.performance_tracking['portfolio_values']) < 2:
            return 0.0, 0.0
        
        current_value = self.performance_tracking['portfolio_values'][-1]
        previous_value = self.performance_tracking['portfolio_values'][-2]
        
        daily_pnl = current_value - previous_value
        daily_pnl_pct = (daily_pnl / previous_value) * 100 if previous_value > 0 else 0
        
        return daily_pnl, daily_pnl_pct
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate exposure by sector."""
        exposure = {}
        
        for position in self.positions.values():
            sector = self._get_symbol_sector(position.symbol)
            exposure[sector] = exposure.get(sector, 0) + position.allocation_pct
        
        return exposure
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)."""
        crypto_sectors = {
            'BTC': 'Store_of_Value',
            'ETH': 'Smart_Contracts',
            'SOL': 'Smart_Contracts', 
            'ADA': 'Smart_Contracts',
            'DOT': 'Interoperability',
            'LINK': 'Oracle',
            'AVAX': 'Smart_Contracts',
            'MATIC': 'Scaling'
        }
        
        base_symbol = symbol.replace('USDT', '').replace('USD', '')
        return crypto_sectors.get(base_symbol, 'Other')
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get estimated volatility for a symbol (mock implementation)."""
        volatility_map = {
            'BTCUSDT': 0.04,
            'ETHUSDT': 0.05,
            'SOLUSDT': 0.08,
            'ADAUSDT': 0.06,
            'DOTUSDT': 0.07
        }
        return volatility_map.get(symbol, 0.06)
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate additional risk from concentration."""
        if not self.positions:
            return 0
        
        max_position = max(pos.allocation_pct for pos in self.positions.values())
        
        if max_position > 0.30:
            return 25
        elif max_position > 0.20:
            return 15
        elif max_position > 0.15:
            return 10
        else:
            return 0
    
    def _calculate_optimal_allocation(self) -> Dict[str, float]:
        """Calculate optimal allocation (simplified equal weight for now)."""
        if not self.positions:
            return {}
        
        equal_weight = 1.0 / len(self.positions)
        return {symbol: equal_weight for symbol in self.positions.keys()}
    
    def _calculate_period_returns(self) -> Dict[str, float]:
        """Calculate returns for different time periods."""
        if len(self.performance_tracking['portfolio_values']) < 2:
            return {}
        
        values = self.performance_tracking['portfolio_values']
        timestamps = self.performance_tracking['timestamps']
        
        returns = {}
        
        # Calculate returns for different periods
        for days, label in [(1, '1D'), (7, '1W'), (30, '1M'), (90, '3M'), (365, '1Y')]:
            if len(values) > days:
                start_value = values[-days-1]
                end_value = values[-1]
                returns[label] = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
        
        return returns
    
    def _generate_risk_recommendations(self, risk_flags: List[str], metrics: PortfolioMetrics) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if metrics.largest_position_pct > 0.20:
            recommendations.append("Consider reducing your largest position to <20% of portfolio")
        
        if len(self.positions) < 3:
            recommendations.append("Consider adding more positions for better diversification")
        
        if metrics.cash_balance / metrics.total_value < 0.10:
            recommendations.append("Maintain at least 10% cash reserves for opportunities and risk management")
        
        if abs(metrics.max_drawdown) > 0.15:
            recommendations.append("Implement stricter stop-losses to limit drawdowns")
        
        return recommendations
    
    def _check_exit_conditions(self, position: Position):
        """Check if position should be closed based on stop loss/take profit."""
        if not position.current_price:
            return
        
        # Stop loss check
        if position.stop_loss and position.current_price <= position.stop_loss:
            logging.info(f"Stop loss triggered for {position.symbol} at ${position.current_price}")
            # In a real implementation, this would trigger an order
        
        # Take profit check
        if position.take_profit and position.current_price >= position.take_profit:
            logging.info(f"Take profit triggered for {position.symbol} at ${position.current_price}")
            # In a real implementation, this would trigger an order
    
    def _record_portfolio_snapshot(self):
        """Record current portfolio state for historical tracking."""
        current_value = self.get_total_portfolio_value()
        
        self.performance_tracking['portfolio_values'].append(current_value)
        self.performance_tracking['timestamps'].append(datetime.now())
        
        # Calculate daily return
        if len(self.performance_tracking['portfolio_values']) > 1:
            prev_value = self.performance_tracking['portfolio_values'][-2]
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            self.performance_tracking['daily_returns'].append(daily_return)
        
        # Keep only last year of data
        max_length = 365
        for key in self.performance_tracking:
            if len(self.performance_tracking[key]) > max_length:
                self.performance_tracking[key] = self.performance_tracking[key][-max_length:]
        
        # Add to portfolio history
        snapshot = {
            'timestamp': datetime.now(),
            'total_value': current_value,
            'cash_balance': self.cash_balance,
            'positions_count': len(self.positions),
            'largest_position': max([pos.allocation_pct for pos in self.positions.values()]) if self.positions else 0
        }
        
        self.portfolio_history.append(snapshot)
        
        # Keep last 1000 snapshots
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]