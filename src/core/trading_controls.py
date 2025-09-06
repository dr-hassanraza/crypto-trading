"""
Trading Window Controls and Circuit Breakers for Crypto Trend Analyzer

This module provides time-based execution controls, circuit breakers,
and system safety mechanisms for the trading system.
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz
from collections import defaultdict
import threading

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async, ErrorCategory


class TradingWindowStatus(Enum):
    """Trading window status."""
    OPEN = "open"
    CLOSED = "closed"
    RESTRICTED = "restricted"
    MAINTENANCE = "maintenance"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking all operations
    HALF_OPEN = "half_open"  # Testing with limited operations


class TradingRestriction(Enum):
    """Types of trading restrictions."""
    NO_NEW_POSITIONS = "no_new_positions"
    REDUCE_ONLY = "reduce_only"
    PAPER_TRADING_ONLY = "paper_trading_only"
    MANUAL_APPROVAL = "manual_approval"
    POSITION_SIZE_LIMITS = "position_size_limits"


@dataclass
class TradingWindow:
    """Trading window configuration."""
    name: str
    start_time: time
    end_time: time
    timezone: str
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    restrictions: List[TradingRestriction]
    priority: int  # Higher priority windows override lower priority


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    name: str
    trigger_condition: str
    threshold_value: float
    time_window_minutes: int
    recovery_time_minutes: int
    max_triggers_per_hour: int
    auto_recovery: bool


class TradingWindowManager:
    """Manages trading windows and time-based restrictions."""
    
    def __init__(self):
        self.trading_windows: List[TradingWindow] = []
        self.current_status = TradingWindowStatus.OPEN
        self.current_restrictions: List[TradingRestriction] = []
        self.manual_overrides: Dict[str, Any] = {}
        
        self._initialize_default_windows()
    
    def _initialize_default_windows(self):
        """Initialize default trading windows."""
        # Main trading window (24/7 for crypto)
        main_window = TradingWindow(
            name="main_trading",
            start_time=time(0, 0),  # 00:00
            end_time=time(23, 59),  # 23:59
            timezone="UTC",
            days_of_week=[0, 1, 2, 3, 4, 5, 6],  # All days
            restrictions=[],
            priority=1
        )
        
        # Restricted weekend window (optional - crypto markets are 24/7)
        weekend_window = TradingWindow(
            name="weekend_restricted",
            start_time=time(0, 0),
            end_time=time(23, 59),
            timezone="UTC",
            days_of_week=[5, 6],  # Saturday, Sunday
            restrictions=[TradingRestriction.POSITION_SIZE_LIMITS],
            priority=2
        )
        
        # High volatility window (during major news/events)
        volatility_window = TradingWindow(
            name="high_volatility",
            start_time=time(0, 0),
            end_time=time(23, 59),
            timezone="UTC",
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            restrictions=[
                TradingRestriction.POSITION_SIZE_LIMITS,
                TradingRestriction.MANUAL_APPROVAL
            ],
            priority=3
        )
        
        self.trading_windows = [main_window, weekend_window, volatility_window]
    
    def add_trading_window(self, window: TradingWindow):
        """Add a new trading window."""
        self.trading_windows.append(window)
        self.trading_windows.sort(key=lambda w: w.priority, reverse=True)
        crypto_logger.logger.info(f"Added trading window: {window.name}")
    
    def get_current_trading_status(self) -> Tuple[TradingWindowStatus, List[TradingRestriction]]:
        """Get current trading status and restrictions."""
        now = datetime.now(pytz.UTC)
        current_time = now.time()
        current_weekday = now.weekday()
        
        # Check manual overrides first
        if 'status_override' in self.manual_overrides:
            override_status = self.manual_overrides['status_override']
            override_restrictions = self.manual_overrides.get('restrictions_override', [])
            return override_status, override_restrictions
        
        # Find applicable trading windows (sorted by priority)
        applicable_restrictions = []
        status = TradingWindowStatus.CLOSED
        
        for window in self.trading_windows:
            if self._is_window_active(window, now):
                status = TradingWindowStatus.OPEN
                applicable_restrictions.extend(window.restrictions)
        
        # Remove duplicates while preserving order
        unique_restrictions = []
        for restriction in applicable_restrictions:
            if restriction not in unique_restrictions:
                unique_restrictions.append(restriction)
        
        self.current_status = status
        self.current_restrictions = unique_restrictions
        
        return status, unique_restrictions
    
    def _is_window_active(self, window: TradingWindow, now: datetime) -> bool:
        """Check if a trading window is currently active."""
        # Convert to window timezone
        window_tz = pytz.timezone(window.timezone)
        window_time = now.astimezone(window_tz)
        current_time = window_time.time()
        current_weekday = window_time.weekday()
        
        # Check day of week
        if current_weekday not in window.days_of_week:
            return False
        
        # Check time range
        if window.start_time <= window.end_time:
            # Normal case (e.g., 09:00 - 17:00)
            return window.start_time <= current_time <= window.end_time
        else:
            # Overnight case (e.g., 22:00 - 06:00)
            return current_time >= window.start_time or current_time <= window.end_time
    
    def can_execute_trade(self, trade_type: str, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Check if a trade can be executed given current restrictions."""
        status, restrictions = self.get_current_trading_status()
        
        # Check if trading is completely closed
        if status == TradingWindowStatus.CLOSED:
            return False, "Trading window is closed"
        
        if status == TradingWindowStatus.MAINTENANCE:
            return False, "System is in maintenance mode"
        
        # Check specific restrictions
        for restriction in restrictions:
            can_execute, reason = self._check_restriction(restriction, trade_type, symbol, position_size)
            if not can_execute:
                return False, f"Restricted: {reason}"
        
        return True, "Trade allowed"
    
    def _check_restriction(self, restriction: TradingRestriction, trade_type: str, 
                          symbol: str, position_size: float) -> Tuple[bool, str]:
        """Check if a specific restriction blocks the trade."""
        if restriction == TradingRestriction.NO_NEW_POSITIONS:
            if trade_type.upper() in ['BUY', 'LONG']:
                return False, "No new positions allowed"
        
        elif restriction == TradingRestriction.REDUCE_ONLY:
            if trade_type.upper() in ['BUY', 'LONG']:
                return False, "Reduce-only mode active"
        
        elif restriction == TradingRestriction.PAPER_TRADING_ONLY:
            return False, "Paper trading only mode"
        
        elif restriction == TradingRestriction.MANUAL_APPROVAL:
            # This would typically check an approval queue/system
            return False, "Manual approval required"
        
        elif restriction == TradingRestriction.POSITION_SIZE_LIMITS:
            # Check position size limits (example: max 50% of normal size)
            max_allowed_size = position_size * 0.5
            if position_size > max_allowed_size:
                return False, f"Position size exceeds limit: {max_allowed_size}"
        
        return True, ""
    
    def set_manual_override(self, status: TradingWindowStatus, 
                           restrictions: List[TradingRestriction] = None,
                           duration_minutes: int = 60):
        """Set manual override for trading status."""
        self.manual_overrides = {
            'status_override': status,
            'restrictions_override': restrictions or [],
            'expires_at': datetime.now() + timedelta(minutes=duration_minutes)
        }
        
        crypto_logger.logger.warning(
            f"Manual override set: {status.value} for {duration_minutes} minutes"
        )
    
    def clear_manual_override(self):
        """Clear manual overrides."""
        self.manual_overrides.clear()
        crypto_logger.logger.info("Manual overrides cleared")
    
    def _cleanup_expired_overrides(self):
        """Clean up expired manual overrides."""
        if 'expires_at' in self.manual_overrides:
            if datetime.now() > self.manual_overrides['expires_at']:
                self.clear_manual_override()


class CircuitBreakerManager:
    """Manages circuit breakers for system protection."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, Dict] = {}
        self.trigger_history: Dict[str, List[datetime]] = defaultdict(list)
        self.current_states: Dict[str, CircuitBreakerState] = {}
        
        self._initialize_default_breakers()
    
    def _initialize_default_breakers(self):
        """Initialize default circuit breakers."""
        default_breakers = [
            CircuitBreakerConfig(
                name="high_loss_rate",
                trigger_condition="consecutive_losses",
                threshold_value=5,  # 5 consecutive losses
                time_window_minutes=60,
                recovery_time_minutes=30,
                max_triggers_per_hour=3,
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="rapid_fire_trading",
                trigger_condition="trades_per_minute",
                threshold_value=10,  # 10 trades per minute
                time_window_minutes=5,
                recovery_time_minutes=15,
                max_triggers_per_hour=2,
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="large_drawdown",
                trigger_condition="portfolio_drawdown",
                threshold_value=0.1,  # 10% drawdown
                time_window_minutes=60,
                recovery_time_minutes=60,
                max_triggers_per_hour=1,
                auto_recovery=False
            ),
            CircuitBreakerConfig(
                name="api_error_rate",
                trigger_condition="api_errors_per_minute",
                threshold_value=5,  # 5 API errors per minute
                time_window_minutes=5,
                recovery_time_minutes=10,
                max_triggers_per_hour=5,
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="extreme_volatility",
                trigger_condition="volatility_spike",
                threshold_value=0.2,  # 20% volatility spike
                time_window_minutes=15,
                recovery_time_minutes=30,
                max_triggers_per_hour=2,
                auto_recovery=True
            )
        ]
        
        for config in default_breakers:
            self.add_circuit_breaker(config)
    
    def add_circuit_breaker(self, config: CircuitBreakerConfig):
        """Add a circuit breaker."""
        self.circuit_breakers[config.name] = {
            'config': config,
            'state': CircuitBreakerState.CLOSED,
            'triggered_at': None,
            'recovery_at': None,
            'trigger_count': 0
        }
        self.current_states[config.name] = CircuitBreakerState.CLOSED
        crypto_logger.logger.info(f"Added circuit breaker: {config.name}")
    
    async def check_circuit_breakers(self, metrics: Dict[str, Any]) -> List[str]:
        """Check all circuit breakers and return list of triggered breakers."""
        triggered_breakers = []
        
        for breaker_name, breaker_data in self.circuit_breakers.items():
            config = breaker_data['config']
            current_state = breaker_data['state']
            
            # Skip if breaker is already open
            if current_state == CircuitBreakerState.OPEN:
                # Check if recovery time has passed
                if breaker_data['recovery_at'] and datetime.now() >= breaker_data['recovery_at']:
                    if config.auto_recovery:
                        await self._transition_to_half_open(breaker_name)
                    else:
                        crypto_logger.logger.warning(
                            f"Circuit breaker {breaker_name} ready for manual reset"
                        )
                continue
            
            # Check trigger condition
            should_trigger = await self._check_trigger_condition(config, metrics)
            
            if should_trigger:
                await self._trigger_circuit_breaker(breaker_name)
                triggered_breakers.append(breaker_name)
        
        return triggered_breakers
    
    async def _check_trigger_condition(self, config: CircuitBreakerConfig, 
                                     metrics: Dict[str, Any]) -> bool:
        """Check if circuit breaker should be triggered."""
        condition = config.trigger_condition
        threshold = config.threshold_value
        
        if condition == "consecutive_losses":
            consecutive_losses = metrics.get('consecutive_losses', 0)
            return consecutive_losses >= threshold
        
        elif condition == "trades_per_minute":
            trades_per_minute = metrics.get('trades_per_minute', 0)
            return trades_per_minute >= threshold
        
        elif condition == "portfolio_drawdown":
            current_drawdown = metrics.get('portfolio_drawdown', 0)
            return current_drawdown >= threshold
        
        elif condition == "api_errors_per_minute":
            api_errors = metrics.get('api_errors_per_minute', 0)
            return api_errors >= threshold
        
        elif condition == "volatility_spike":
            volatility_spike = metrics.get('volatility_spike', 0)
            return volatility_spike >= threshold
        
        return False
    
    async def _trigger_circuit_breaker(self, breaker_name: str):
        """Trigger a circuit breaker."""
        breaker_data = self.circuit_breakers[breaker_name]
        config = breaker_data['config']
        
        # Check if we've exceeded max triggers per hour
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean old triggers
        self.trigger_history[breaker_name] = [
            trigger_time for trigger_time in self.trigger_history[breaker_name]
            if trigger_time > one_hour_ago
        ]
        
        if len(self.trigger_history[breaker_name]) >= config.max_triggers_per_hour:
            crypto_logger.logger.critical(
                f"Circuit breaker {breaker_name} reached max triggers per hour. "
                f"Manual intervention required."
            )
            return
        
        # Trigger the breaker
        breaker_data['state'] = CircuitBreakerState.OPEN
        breaker_data['triggered_at'] = now
        breaker_data['recovery_at'] = now + timedelta(minutes=config.recovery_time_minutes)
        breaker_data['trigger_count'] += 1
        
        self.current_states[breaker_name] = CircuitBreakerState.OPEN
        self.trigger_history[breaker_name].append(now)
        
        crypto_logger.logger.error(
            f"🚨 Circuit breaker TRIGGERED: {breaker_name} "
            f"(recovery in {config.recovery_time_minutes} minutes)"
        )
        
        # Log detailed trigger information
        await handle_error_async(
            Exception(f"Circuit breaker triggered: {breaker_name}"),
            {'component': 'circuit_breaker', 'breaker': breaker_name}
        )
    
    async def _transition_to_half_open(self, breaker_name: str):
        """Transition circuit breaker to half-open state."""
        breaker_data = self.circuit_breakers[breaker_name]
        breaker_data['state'] = CircuitBreakerState.HALF_OPEN
        self.current_states[breaker_name] = CircuitBreakerState.HALF_OPEN
        
        crypto_logger.logger.info(
            f"Circuit breaker {breaker_name} transitioned to HALF-OPEN"
        )
        
        # Schedule transition to closed after test period
        asyncio.create_task(self._test_half_open_state(breaker_name))
    
    async def _test_half_open_state(self, breaker_name: str):
        """Test half-open state and decide whether to close or reopen."""
        test_duration = 300  # 5 minutes test period
        await asyncio.sleep(test_duration)
        
        # For simplicity, auto-close after test period
        # In a real implementation, you'd check if conditions are stable
        breaker_data = self.circuit_breakers[breaker_name]
        if breaker_data['state'] == CircuitBreakerState.HALF_OPEN:
            breaker_data['state'] = CircuitBreakerState.CLOSED
            self.current_states[breaker_name] = CircuitBreakerState.CLOSED
            
            crypto_logger.logger.info(
                f"Circuit breaker {breaker_name} reset to CLOSED"
            )
    
    def is_circuit_breaker_open(self, breaker_name: str = None) -> bool:
        """Check if any or specific circuit breaker is open."""
        if breaker_name:
            return self.current_states.get(breaker_name) == CircuitBreakerState.OPEN
        
        return any(
            state == CircuitBreakerState.OPEN 
            for state in self.current_states.values()
        )
    
    def get_open_circuit_breakers(self) -> List[str]:
        """Get list of open circuit breakers."""
        return [
            name for name, state in self.current_states.items()
            if state == CircuitBreakerState.OPEN
        ]
    
    def manually_reset_circuit_breaker(self, breaker_name: str):
        """Manually reset a circuit breaker."""
        if breaker_name in self.circuit_breakers:
            breaker_data = self.circuit_breakers[breaker_name]
            breaker_data['state'] = CircuitBreakerState.CLOSED
            breaker_data['triggered_at'] = None
            breaker_data['recovery_at'] = None
            
            self.current_states[breaker_name] = CircuitBreakerState.CLOSED
            
            crypto_logger.logger.info(
                f"Circuit breaker {breaker_name} manually reset"
            )
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {}
        
        for breaker_name, breaker_data in self.circuit_breakers.items():
            status[breaker_name] = {
                'state': breaker_data['state'].value,
                'trigger_count': breaker_data['trigger_count'],
                'triggered_at': breaker_data['triggered_at'],
                'recovery_at': breaker_data['recovery_at'],
                'recent_triggers': len(self.trigger_history[breaker_name])
            }
        
        return status


class TradingControlSystem:
    """Main trading control system combining windows and circuit breakers."""
    
    def __init__(self):
        self.window_manager = TradingWindowManager()
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.system_metrics = {}
        self._monitoring_task = None
    
    async def initialize(self):
        """Initialize the trading control system."""
        crypto_logger.logger.info("🛡️ Trading Control System initialized")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def can_execute_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a trade can be executed."""
        # Check trading windows
        can_trade, window_reason = self.window_manager.can_execute_trade(
            trade_data.get('type', 'buy'),
            trade_data.get('symbol', ''),
            trade_data.get('size', 0)
        )
        
        if not can_trade:
            return False, f"Trading window restriction: {window_reason}"
        
        # Check circuit breakers
        if self.circuit_breaker_manager.is_circuit_breaker_open():
            open_breakers = self.circuit_breaker_manager.get_open_circuit_breakers()
            return False, f"Circuit breaker(s) active: {', '.join(open_breakers)}"
        
        return True, "Trade execution allowed"
    
    async def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics for monitoring."""
        self.system_metrics.update(metrics)
        
        # Check circuit breakers
        triggered_breakers = await self.circuit_breaker_manager.check_circuit_breakers(metrics)
        
        if triggered_breakers:
            crypto_logger.logger.error(
                f"Circuit breakers triggered: {', '.join(triggered_breakers)}"
            )
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                # Clean up expired overrides
                self.window_manager._cleanup_expired_overrides()
                
                # Update current status
                status, restrictions = self.window_manager.get_current_trading_status()
                
                # Log status changes
                if status != self.window_manager.current_status:
                    crypto_logger.logger.info(f"Trading status changed to: {status.value}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await handle_error_async(e, {'component': 'trading_control_monitoring'})
                await asyncio.sleep(300)  # Wait longer on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        window_status, restrictions = self.window_manager.get_current_trading_status()
        circuit_status = self.circuit_breaker_manager.get_circuit_breaker_status()
        
        return {
            'trading_window_status': window_status.value,
            'active_restrictions': [r.value for r in restrictions],
            'circuit_breakers': circuit_status,
            'system_metrics': self.system_metrics,
            'timestamp': datetime.now()
        }
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown of all trading."""
        self.window_manager.set_manual_override(
            TradingWindowStatus.CLOSED,
            [],
            duration_minutes=1440  # 24 hours
        )
        
        crypto_logger.logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")
    
    def stop(self):
        """Stop the trading control system."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        crypto_logger.logger.info("Trading Control System stopped")


# Global trading control system instance
trading_control_system = TradingControlSystem()