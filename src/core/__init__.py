"""
Core System Components for Crypto Trend Analyzer

This module provides the core infrastructure components that implement
the enhanced algorithmic trading system architecture including:

- Error Handling Engine
- Validation Models (Guardrails)
- Black Box Processing (ML + Probabilistic Engine)
- Trading Window Controls & Circuit Breakers
- Control & Monitoring Center
"""

from .error_handling_engine import (
    ErrorHandlingEngine,
    ErrorSeverity,
    ErrorCategory,
    ErrorEvent,
    error_handler,
    handle_error_async,
    handle_error_sync
)

from .validation_models import (
    ValidationResult,
    ValidationCategory,
    ValidationReport,
    SignalQualityValidator,
    RiskAssessmentValidator,
    MarketConditionsValidator,
    ValidationEngine,
    validation_engine
)

from .black_box_processing import (
    TradingDecision,
    DecisionConfidence,
    ProbabilisticOutput,
    ModelPrediction,
    EnsembleMLEngine,
    ProbabilisticEngine,
    BlackBoxProcessor,
    black_box_processor
)

from .trading_controls import (
    TradingWindowStatus,
    CircuitBreakerState,
    TradingRestriction,
    TradingWindow,
    CircuitBreakerConfig,
    TradingWindowManager,
    CircuitBreakerManager,
    TradingControlSystem,
    trading_control_system
)

from .control_monitoring_center import (
    SystemHealth,
    AlertPriority,
    SystemAlert,
    SystemMetrics,
    PerformanceMonitor,
    AlertManager,
    ControlOverrideManager,
    ControlMonitoringCenter,
    control_center
)

__all__ = [
    # Error Handling
    'ErrorHandlingEngine',
    'ErrorSeverity', 
    'ErrorCategory',
    'ErrorEvent',
    'error_handler',
    'handle_error_async',
    'handle_error_sync',
    
    # Validation Models
    'ValidationResult',
    'ValidationCategory',
    'ValidationReport',
    'SignalQualityValidator',
    'RiskAssessmentValidator',
    'MarketConditionsValidator',
    'ValidationEngine',
    'validation_engine',
    
    # Black Box Processing
    'TradingDecision',
    'DecisionConfidence',
    'ProbabilisticOutput',
    'ModelPrediction',
    'EnsembleMLEngine',
    'ProbabilisticEngine',
    'BlackBoxProcessor',
    'black_box_processor',
    
    # Trading Controls
    'TradingWindowStatus',
    'CircuitBreakerState',
    'TradingRestriction',
    'TradingWindow',
    'CircuitBreakerConfig',
    'TradingWindowManager',
    'CircuitBreakerManager',
    'TradingControlSystem',
    'trading_control_system',
    
    # Control & Monitoring
    'SystemHealth',
    'AlertPriority',
    'SystemAlert',
    'SystemMetrics',
    'PerformanceMonitor',
    'AlertManager',
    'ControlOverrideManager',
    'ControlMonitoringCenter',
    'control_center'
]