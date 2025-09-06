"""
Advanced Analytics Suite for Crypto Trend Analyzer

Comprehensive analytics system featuring:
- HDBSCAN clustering with strict performance rules
- Advanced preprocessing and feature selection pipeline
- Metrics Engine API integration
- Bayesian probability framework
- Market classification and opportunity identification
- Sub-500ms performance optimization

All components work together to provide real-time market analysis
with rigorous performance constraints and intelligent decision making.
"""

from .clustering_engine import (
    clustering_engine,
    ClusteringConfig,
    ClusterResult,
    MarketCluster,
    HighPerformanceHDBSCAN,
    PerformanceOptimizedPreprocessor,
    ClusteringEngine
)

from .feature_pipeline import (
    feature_pipeline,
    FeaturePipelineConfig,
    FeatureImportance,
    PipelineResult,
    TechnicalIndicatorEngine,
    FeatureSelector,
    AdvancedFeaturePipeline
)

from .metrics_engine import (
    metrics_engine,
    MetricsConfig,
    RealTimeMetrics,
    MarketState,
    HighPerformanceAPIClient,
    TechnicalMetricsCalculator,
    AdvancedMetricsEngine
)

from .bayesian_framework import (
    bayesian_framework,
    BayesianConfig,
    PriorDistribution,
    Evidence,
    PosteriorDistribution,
    BayesianInferenceResult,
    BayesianPriorEngine,
    BayesianLikelihoodEngine,
    BayesianInferenceEngine,
    BayesianDecisionEngine,
    AdvancedBayesianFramework
)

from .market_classification import (
    market_classification_engine,
    MarketRegime,
    OpportunityType,
    MarketClassification,
    TradingOpportunity,
    MarketOpportunityAnalysis,
    MarketRegimeClassifier,
    OpportunityDetector,
    AdvancedMarketClassificationEngine
)

from .performance_optimizer import (
    performance_optimizer,
    PerformanceConfig,
    PerformanceMetrics,
    OptimizationRecommendation,
    IntelligentCache,
    PerformanceProfiler,
    AdaptiveOptimizer,
    AdvancedPerformanceOptimizer,
    performance_monitor,
    start_timing,
    record_component_time,
    finish_timing,
    optimize_async_operation,
    get_performance_stats
)

__version__ = "1.0.0"

__all__ = [
    # Clustering
    'clustering_engine',
    'ClusteringConfig',
    'ClusterResult',
    'MarketCluster',
    'HighPerformanceHDBSCAN',
    'PerformanceOptimizedPreprocessor',
    'ClusteringEngine',
    
    # Feature Pipeline
    'feature_pipeline',
    'FeaturePipelineConfig', 
    'FeatureImportance',
    'PipelineResult',
    'TechnicalIndicatorEngine',
    'FeatureSelector',
    'AdvancedFeaturePipeline',
    
    # Metrics Engine
    'metrics_engine',
    'MetricsConfig',
    'RealTimeMetrics',
    'MarketState',
    'HighPerformanceAPIClient',
    'TechnicalMetricsCalculator',
    'AdvancedMetricsEngine',
    
    # Bayesian Framework
    'bayesian_framework',
    'BayesianConfig',
    'PriorDistribution',
    'Evidence',
    'PosteriorDistribution',
    'BayesianInferenceResult',
    'BayesianPriorEngine',
    'BayesianLikelihoodEngine',
    'BayesianInferenceEngine',
    'BayesianDecisionEngine',
    'AdvancedBayesianFramework',
    
    # Market Classification
    'market_classification_engine',
    'MarketRegime',
    'OpportunityType',
    'MarketClassification',
    'TradingOpportunity',
    'MarketOpportunityAnalysis',
    'MarketRegimeClassifier',
    'OpportunityDetector',
    'AdvancedMarketClassificationEngine',
    
    # Performance Optimization
    'performance_optimizer',
    'PerformanceConfig',
    'PerformanceMetrics',
    'OptimizationRecommendation',
    'IntelligentCache',
    'PerformanceProfiler',
    'AdaptiveOptimizer',
    'AdvancedPerformanceOptimizer',
    'performance_monitor',
    'start_timing',
    'record_component_time',
    'finish_timing',
    'optimize_async_operation',
    'get_performance_stats'
]


# Integrated Analytics API
async def comprehensive_market_analysis(market_data: dict, symbol: str = None) -> dict:
    """
    Perform comprehensive market analysis using all advanced analytics components.
    
    This is the main entry point that orchestrates all components to provide
    a complete market analysis with sub-500ms performance guarantee.
    
    Args:
        market_data: Market data dictionary containing OHLCV, technical analysis, etc.
        symbol: Trading symbol (optional, will be extracted from market_data if not provided)
    
    Returns:
        Complete analysis result containing:
        - Market classification and regime identification
        - Trading opportunities with risk assessment
        - Bayesian probability analysis
        - Clustering insights
        - Performance metrics
    """
    from .performance_optimizer import start_timing, finish_timing
    
    # Start performance tracking
    operation_id = start_timing("comprehensive_market_analysis")
    
    try:
        # Ensure symbol is available
        if not symbol:
            symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Ensure market_data has symbol
        if 'symbol' not in market_data:
            market_data['symbol'] = symbol
        
        # Run comprehensive analysis through market classification engine
        # This orchestrates all other components internally
        analysis_result = await market_classification_engine.analyze_market_comprehensively(market_data)
        
        # Convert to dictionary format for API response
        result = {
            'symbol': analysis_result.symbol,
            'timestamp': analysis_result.timestamp.isoformat(),
            'market_classification': {
                'primary_regime': analysis_result.market_classification.primary_regime.value,
                'regime_confidence': analysis_result.market_classification.regime_confidence,
                'regime_probabilities': {
                    k.value: v for k, v in analysis_result.market_classification.regime_probabilities.items()
                },
                'trend_strength': analysis_result.market_classification.trend_strength,
                'momentum_score': analysis_result.market_classification.momentum_score,
                'volatility_percentile': analysis_result.market_classification.volatility_percentile,
                'cluster_id': analysis_result.market_classification.cluster_id,
                'outlier_score': analysis_result.market_classification.outlier_score
            },
            'trading_opportunities': [
                {
                    'type': opp.opportunity_type.value,
                    'score': opp.opportunity_score,
                    'confidence': opp.confidence_level,
                    'suggested_action': opp.suggested_action,
                    'entry_price': opp.entry_price,
                    'stop_loss': opp.stop_loss,
                    'take_profit': opp.take_profit,
                    'position_size_recommendation': opp.position_size_recommendation,
                    'risk_reward_ratio': opp.risk_reward_ratio,
                    'win_probability': opp.win_probability,
                    'technical_signals': opp.technical_signals,
                    'urgency_score': opp.urgency_score
                }
                for opp in analysis_result.identified_opportunities
            ],
            'overall_assessment': {
                'market_score': analysis_result.overall_market_score,
                'risk_adjusted_score': analysis_result.risk_adjusted_score,
                'recommendation': analysis_result.recommendation,
                'analysis_confidence': analysis_result.analysis_confidence
            },
            'performance_metrics': {
                'processing_time_ms': analysis_result.processing_time_ms,
                'target_met': analysis_result.processing_time_ms < 500
            }
        }
        
        # Record successful completion
        finish_timing(operation_id, success=True)
        
        return result
        
    except Exception as e:
        # Record failed completion
        finish_timing(operation_id, success=False, error=str(e))
        
        # Return error result
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'market_classification': None,
            'trading_opportunities': [],
            'overall_assessment': {
                'market_score': 0.0,
                'risk_adjusted_score': 0.0,
                'recommendation': 'avoid',
                'analysis_confidence': 0.0
            },
            'performance_metrics': {
                'processing_time_ms': 0,
                'target_met': False
            }
        }


async def quick_market_analysis(market_data: dict, symbol: str = None) -> dict:
    """
    Perform quick market analysis optimized for ultra-low latency (<200ms).
    
    Uses cached results and simplified algorithms for speed.
    """
    from .performance_optimizer import start_timing, finish_timing
    
    operation_id = start_timing("quick_market_analysis")
    
    try:
        if not symbol:
            symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Quick Bayesian analysis only
        bayesian_result = await bayesian_framework.analyze_market_bayesian(market_data)
        
        # Simple opportunity detection
        opportunities = []
        if bayesian_result.decision in ['buy', 'strong_buy']:
            opportunities.append({
                'type': 'momentum_long',
                'score': bayesian_result.confidence_score,
                'suggested_action': 'buy',
                'confidence': bayesian_result.confidence_score
            })
        elif bayesian_result.decision in ['sell', 'strong_sell']:
            opportunities.append({
                'type': 'momentum_short',
                'score': bayesian_result.confidence_score,
                'suggested_action': 'sell',
                'confidence': bayesian_result.confidence_score
            })
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'quick_analysis': True,
            'bayesian_decision': bayesian_result.decision,
            'confidence_score': bayesian_result.confidence_score,
            'expected_value': bayesian_result.expected_value,
            'opportunities': opportunities,
            'processing_time_ms': bayesian_result.processing_time_ms
        }
        
        finish_timing(operation_id, success=True)
        return result
        
    except Exception as e:
        finish_timing(operation_id, success=False, error=str(e))
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'quick_analysis': True
        }


def get_analytics_system_status() -> dict:
    """Get comprehensive status of all analytics components."""
    return {
        'clustering_engine': clustering_engine.get_performance_stats(),
        'feature_pipeline': feature_pipeline.get_performance_stats(),
        'metrics_engine': metrics_engine.get_performance_stats(),
        'bayesian_framework': bayesian_framework.get_framework_statistics(),
        'market_classification_engine': market_classification_engine.get_engine_statistics(),
        'performance_optimizer': performance_optimizer.get_performance_summary(),
        'overall_health': 'optimal' if all([
            clustering_engine.get_performance_stats().get('performance_target_met', True),
            feature_pipeline.get_performance_stats().get('performance_target_success_rate', 1.0) > 0.8,
            bayesian_framework.get_framework_statistics().get('performance_target_met', True)
        ]) else 'degraded'
    }


# Export the main API functions
__all__.extend([
    'comprehensive_market_analysis',
    'quick_market_analysis', 
    'get_analytics_system_status'
])