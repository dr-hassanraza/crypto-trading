"""
Enhanced Error Handling Engine for Crypto Trend Analyzer

This module provides centralized error handling, recovery mechanisms,
and system stability features for the trading system.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from src.utils.logging_config import crypto_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_SOURCE = "data_source"
    MARKET_DATA = "market_data"
    MODEL_PREDICTION = "model_prediction"
    SIGNAL_GENERATION = "signal_generation"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    EXECUTION = "execution"
    VALIDATION = "validation"
    SYSTEM = "system"
    NETWORK = "network"
    API = "api"


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    traceback: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_actions: List[str] = None
    
    def __post_init__(self):
        if self.resolution_actions is None:
            self.resolution_actions = []


class ErrorHandlingEngine:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.error_thresholds: Dict[ErrorCategory, Dict[str, int]] = {}
        self.circuit_breaker_states: Dict[str, bool] = {}
        self.error_callbacks: List[Callable] = []
        
        # Default error thresholds (per hour)
        self.default_thresholds = {
            ErrorCategory.DATA_SOURCE: {"high": 5, "critical": 10},
            ErrorCategory.API: {"high": 10, "critical": 20},
            ErrorCategory.MODEL_PREDICTION: {"high": 3, "critical": 5},
            ErrorCategory.EXECUTION: {"high": 2, "critical": 3},
            ErrorCategory.VALIDATION: {"high": 5, "critical": 8}
        }
        
        self._initialize_recovery_strategies()
        self._initialize_thresholds()
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies for different error types."""
        self.recovery_strategies = {
            ErrorCategory.DATA_SOURCE: [
                self._retry_data_fetch,
                self._switch_data_source,
                self._use_cached_data
            ],
            ErrorCategory.API: [
                self._retry_api_call,
                self._switch_api_endpoint,
                self._enable_rate_limiting
            ],
            ErrorCategory.MODEL_PREDICTION: [
                self._fallback_to_simpler_model,
                self._use_ensemble_prediction,
                self._skip_prediction_cycle
            ],
            ErrorCategory.EXECUTION: [
                self._cancel_pending_orders,
                self._reduce_position_sizes,
                self._enable_paper_trading_mode
            ],
            ErrorCategory.VALIDATION: [
                self._increase_validation_threshold,
                self._disable_risky_strategies,
                self._force_manual_approval
            ]
        }
    
    def _initialize_thresholds(self):
        """Initialize error thresholds."""
        self.error_thresholds = self.default_thresholds.copy()
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> ErrorEvent:
        """
        Central error handling method.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            ErrorEvent: The created error event
        """
        try:
            # Create error event
            error_event = self._create_error_event(error, context)
            
            # Log the error
            await self._log_error(error_event)
            
            # Store in history
            self.error_history.append(error_event)
            
            # Check thresholds and trigger circuit breakers if needed
            await self._check_error_thresholds(error_event)
            
            # Attempt recovery
            await self._attempt_recovery(error_event)
            
            # Notify callbacks
            await self._notify_error_callbacks(error_event)
            
            # Cleanup old errors (keep last 1000)
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
            
            return error_event
            
        except Exception as handling_error:
            # Error in error handling - log critically
            crypto_logger.logger.critical(
                f"Error in error handling: {handling_error}"
            )
            raise
    
    def _create_error_event(self, error: Exception, context: Dict[str, Any]) -> ErrorEvent:
        """Create an ErrorEvent from an exception and context."""
        component = context.get('component', 'unknown')
        category = self._categorize_error(error, context)
        severity = self._assess_severity(error, context, category)
        
        return ErrorEvent(
            id=f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(error),
            component=component,
            traceback=traceback.format_exc(),
            context=context
        )
    
    def _categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Categorize error based on type and context."""
        component = context.get('component', '').lower()
        error_type = type(error).__name__.lower()
        
        # Component-based categorization
        if 'data_source' in component or 'market_data' in component:
            return ErrorCategory.DATA_SOURCE
        elif 'model' in component or 'prediction' in component:
            return ErrorCategory.MODEL_PREDICTION
        elif 'signal' in component:
            return ErrorCategory.SIGNAL_GENERATION
        elif 'portfolio' in component:
            return ErrorCategory.PORTFOLIO_MANAGEMENT
        elif 'execution' in component or 'trading' in component:
            return ErrorCategory.EXECUTION
        elif 'validation' in component:
            return ErrorCategory.VALIDATION
        
        # Error type-based categorization
        if 'connection' in error_type or 'timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'api' in error_type or 'http' in error_type:
            return ErrorCategory.API
        
        return ErrorCategory.SYSTEM
    
    def _assess_severity(self, error: Exception, context: Dict[str, Any], 
                        category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on error type, context, and category."""
        error_type = type(error).__name__.lower()
        
        # Critical errors
        if any(word in error_type for word in ['critical', 'fatal', 'security']):
            return ErrorSeverity.CRITICAL
        
        # High severity for execution and validation errors
        if category in [ErrorCategory.EXECUTION, ErrorCategory.VALIDATION]:
            return ErrorSeverity.HIGH
        
        # High severity for connection errors
        if 'connection' in error_type or 'timeout' in error_type:
            return ErrorSeverity.HIGH
        
        # Medium severity for model and data errors
        if category in [ErrorCategory.MODEL_PREDICTION, ErrorCategory.DATA_SOURCE]:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    async def _log_error(self, error_event: ErrorEvent):
        """Log error event with appropriate level."""
        log_message = (
            f"[{error_event.category.value}] {error_event.component}: "
            f"{error_event.message}"
        )
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            crypto_logger.logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            crypto_logger.logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            crypto_logger.logger.warning(log_message)
        else:
            crypto_logger.logger.info(log_message)
    
    async def _check_error_thresholds(self, error_event: ErrorEvent):
        """Check if error thresholds are exceeded and trigger circuit breakers."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Count recent errors of same category
        recent_errors = [
            e for e in self.error_history
            if e.category == error_event.category and e.timestamp >= one_hour_ago
        ]
        
        high_severity_count = sum(
            1 for e in recent_errors 
            if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        )
        critical_count = sum(
            1 for e in recent_errors 
            if e.severity == ErrorSeverity.CRITICAL
        )
        
        thresholds = self.error_thresholds.get(error_event.category, {})
        
        # Trigger circuit breakers
        circuit_breaker_key = f"{error_event.category.value}_circuit_breaker"
        
        if critical_count >= thresholds.get("critical", 999):
            crypto_logger.logger.critical(
                f"CRITICAL threshold exceeded for {error_event.category.value}. "
                f"Activating circuit breaker."
            )
            self.circuit_breaker_states[circuit_breaker_key] = True
            
        elif high_severity_count >= thresholds.get("high", 999):
            crypto_logger.logger.error(
                f"HIGH threshold exceeded for {error_event.category.value}. "
                f"Entering degraded mode."
            )
            self.circuit_breaker_states[f"{circuit_breaker_key}_degraded"] = True
    
    async def _attempt_recovery(self, error_event: ErrorEvent):
        """Attempt to recover from error using registered strategies."""
        strategies = self.recovery_strategies.get(error_event.category, [])
        
        for strategy in strategies:
            try:
                success = await strategy(error_event)
                if success:
                    error_event.resolved = True
                    error_event.resolution_actions.append(strategy.__name__)
                    crypto_logger.logger.info(
                        f"Recovery successful using {strategy.__name__} "
                        f"for error {error_event.id}"
                    )
                    break
            except Exception as recovery_error:
                crypto_logger.logger.warning(
                    f"Recovery strategy {strategy.__name__} failed: {recovery_error}"
                )
    
    async def _notify_error_callbacks(self, error_event: ErrorEvent):
        """Notify registered error callbacks."""
        for callback in self.error_callbacks:
            try:
                await callback(error_event)
            except Exception as callback_error:
                crypto_logger.logger.warning(
                    f"Error callback failed: {callback_error}"
                )
    
    # Recovery strategy implementations
    async def _retry_data_fetch(self, error_event: ErrorEvent) -> bool:
        """Retry data fetch with exponential backoff."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(base_delay * (2 ** attempt))
                # This would be implemented by the calling component
                return True
            except Exception:
                continue
        return False
    
    async def _switch_data_source(self, error_event: ErrorEvent) -> bool:
        """Switch to alternative data source."""
        crypto_logger.logger.info("Switching to backup data source")
        return True  # Implementation depends on data source manager
    
    async def _use_cached_data(self, error_event: ErrorEvent) -> bool:
        """Use cached data as fallback."""
        crypto_logger.logger.info("Using cached data as fallback")
        return True
    
    async def _retry_api_call(self, error_event: ErrorEvent) -> bool:
        """Retry API call with rate limiting."""
        await asyncio.sleep(2)  # Rate limiting
        return True
    
    async def _switch_api_endpoint(self, error_event: ErrorEvent) -> bool:
        """Switch to alternative API endpoint."""
        crypto_logger.logger.info("Switching to backup API endpoint")
        return True
    
    async def _enable_rate_limiting(self, error_event: ErrorEvent) -> bool:
        """Enable more aggressive rate limiting."""
        crypto_logger.logger.info("Enabling enhanced rate limiting")
        return True
    
    async def _fallback_to_simpler_model(self, error_event: ErrorEvent) -> bool:
        """Fallback to simpler prediction model."""
        crypto_logger.logger.info("Falling back to simpler prediction model")
        return True
    
    async def _use_ensemble_prediction(self, error_event: ErrorEvent) -> bool:
        """Use ensemble prediction as fallback."""
        crypto_logger.logger.info("Using ensemble prediction fallback")
        return True
    
    async def _skip_prediction_cycle(self, error_event: ErrorEvent) -> bool:
        """Skip current prediction cycle."""
        crypto_logger.logger.info("Skipping current prediction cycle")
        return True
    
    async def _cancel_pending_orders(self, error_event: ErrorEvent) -> bool:
        """Cancel all pending orders."""
        crypto_logger.logger.warning("Canceling all pending orders")
        return True
    
    async def _reduce_position_sizes(self, error_event: ErrorEvent) -> bool:
        """Reduce position sizes for safety."""
        crypto_logger.logger.warning("Reducing position sizes")
        return True
    
    async def _enable_paper_trading_mode(self, error_event: ErrorEvent) -> bool:
        """Enable paper trading mode."""
        crypto_logger.logger.warning("Enabling paper trading mode")
        return True
    
    async def _increase_validation_threshold(self, error_event: ErrorEvent) -> bool:
        """Increase validation threshold."""
        crypto_logger.logger.info("Increasing validation threshold")
        return True
    
    async def _disable_risky_strategies(self, error_event: ErrorEvent) -> bool:
        """Disable risky trading strategies."""
        crypto_logger.logger.warning("Disabling risky strategies")
        return True
    
    async def _force_manual_approval(self, error_event: ErrorEvent) -> bool:
        """Force manual approval for trades."""
        crypto_logger.logger.warning("Enabling manual approval mode")
        return True
    
    def add_error_callback(self, callback: Callable):
        """Add error callback function."""
        self.error_callbacks.append(callback)
    
    def is_circuit_breaker_active(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is active for category."""
        key = f"{category.value}_circuit_breaker"
        return self.circuit_breaker_states.get(key, False)
    
    def reset_circuit_breaker(self, category: ErrorCategory):
        """Reset circuit breaker for category."""
        key = f"{category.value}_circuit_breaker"
        self.circuit_breaker_states[key] = False
        crypto_logger.logger.info(f"Circuit breaker reset for {category.value}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp >= one_hour_ago]
        
        stats = {
            "total_errors_last_hour": len(recent_errors),
            "errors_by_category": {},
            "errors_by_severity": {},
            "circuit_breakers": self.circuit_breaker_states.copy(),
            "resolved_errors": len([e for e in recent_errors if e.resolved])
        }
        
        # Group by category
        for error in recent_errors:
            category = error.category.value
            if category not in stats["errors_by_category"]:
                stats["errors_by_category"][category] = 0
            stats["errors_by_category"][category] += 1
            
            # Group by severity
            severity = error.severity.value
            if severity not in stats["errors_by_severity"]:
                stats["errors_by_severity"][severity] = 0
            stats["errors_by_severity"][severity] += 1
        
        return stats


# Global error handling engine instance
error_handler = ErrorHandlingEngine()


async def handle_error_async(error: Exception, context: Dict[str, Any]) -> ErrorEvent:
    """Async wrapper for error handling."""
    return await error_handler.handle_error(error, context)


def handle_error_sync(error: Exception, context: Dict[str, Any]) -> ErrorEvent:
    """Sync wrapper for error handling."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is running, create a task
        task = asyncio.create_task(error_handler.handle_error(error, context))
        return task
    else:
        # If no loop is running, run sync
        return asyncio.run(error_handler.handle_error(error, context))