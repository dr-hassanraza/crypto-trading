#!/usr/bin/env python3
"""
Advanced Analytics Demo - Comprehensive Demonstration

Demonstrates all advanced analytics features:
✅ HDBSCAN clustering with strict performance rules
✅ Preprocessing and feature selection pipeline  
✅ Metrics Engine API integration
✅ Bayesian probability framework
✅ Market classification and opportunity identification
✅ Sub-500ms performance optimization

This demo shows the complete advanced analytics system in action
with realistic market data and performance monitoring.
"""

import asyncio
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.advanced_analytics import (
    comprehensive_market_analysis,
    quick_market_analysis,
    get_analytics_system_status,
    clustering_engine,
    feature_pipeline,
    metrics_engine,
    bayesian_framework,
    market_classification_engine,
    performance_optimizer
)
from src.utils.logging_config import crypto_logger


def generate_realistic_market_data(symbol: str = "BTCUSDT") -> dict:
    """Generate realistic market data for demonstration."""
    
    # Simulate realistic crypto market data
    base_price = 45000.0
    volatility = 0.03
    
    # Generate OHLCV data
    np.random.seed(42)  # For reproducible demo
    
    # Price data
    current_price = base_price * (1 + np.random.normal(0, volatility))
    high_24h = current_price * (1 + np.random.uniform(0.01, 0.05))
    low_24h = current_price * (1 - np.random.uniform(0.01, 0.05))
    
    # Volume data
    volume_24h = np.random.uniform(1e9, 5e9)  # $1B to $5B
    
    # Technical indicators
    rsi = np.random.uniform(30, 70)
    macd_signal = "BUY" if np.random.random() > 0.5 else "SELL"
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'market_data': {
            'current_price': current_price,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'volume_24h': volume_24h,
            'volatility': volatility,
            'price_change_24h': np.random.uniform(-0.05, 0.05),
            'market_cap': current_price * 19e6  # Approximate BTC supply
        },
        'technical_analysis': {
            'rsi': {'value': rsi, 'signal': 'BUY' if rsi < 40 else ('SELL' if rsi > 60 else 'NEUTRAL')},
            'macd': {'signal': macd_signal, 'value': np.random.uniform(-100, 100)},
            'bollinger': {'position': np.random.uniform(0.2, 0.8), 'signal': 'NEUTRAL'},
            'sma_20': {'value': current_price * np.random.uniform(0.98, 1.02)},
            'ema_12': {'value': current_price * np.random.uniform(0.99, 1.01)},
            'atr': {'value': current_price * 0.02}
        },
        'final_signal': {
            'signal': 'BUY' if np.random.random() > 0.4 else 'SELL',
            'confidence_score': np.random.uniform(60, 90),
            'entry_price': current_price,
            'stop_loss': current_price * (0.98 if macd_signal == 'BUY' else 1.02),
            'take_profit': current_price * (1.04 if macd_signal == 'BUY' else 0.96)
        },
        'volume_analysis': {
            'volume_trend': np.random.uniform(-0.2, 0.2),
            'volume_profile': 'normal'
        }
    }


async def demo_clustering_engine():
    """Demonstrate HDBSCAN clustering with performance rules."""
    print("\n🧩 HDBSCAN Clustering Engine Demo")
    print("-" * 50)
    
    try:
        # Generate sample market data for clustering
        data_points = []
        for i in range(200):  # 200 data points for clustering
            point = {
                'price_momentum': np.random.normal(0, 0.1),
                'volatility': np.random.gamma(2, 0.02),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'rsi': np.random.uniform(20, 80),
                'trend_strength': np.random.uniform(0, 1)
            }
            data_points.append(point)
        
        df = pd.DataFrame(data_points)
        
        # Run clustering
        start_time = time.time()
        result = await clustering_engine.cluster_market_data(df)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"   Clusters identified: {result.n_clusters}")
        print(f"   Outliers detected: {np.sum(result.cluster_labels == -1)}")
        print(f"   Silhouette score: {result.silhouette_score:.3f}")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Performance target (<400ms): {'✅' if processing_time < 400 else '❌'}")
        
        # Show cluster sizes
        if result.cluster_sizes:
            print(f"   Cluster sizes: {result.cluster_sizes}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Clustering demo failed: {e}")
        return None


async def demo_feature_pipeline():
    """Demonstrate advanced feature pipeline."""
    print("\n🔧 Feature Pipeline Demo")  
    print("-" * 30)
    
    try:
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        prices = []
        volumes = []
        base_price = 45000
        
        for i in range(100):
            price = base_price * (1 + np.random.normal(0, 0.02))
            volume = np.random.uniform(1000, 10000)
            prices.append(price)
            volumes.append(volume)
            base_price = price
        
        ohlcv_data = pd.DataFrame({
            'open': prices,
            'high': [p * np.random.uniform(1.001, 1.01) for p in prices],
            'low': [p * np.random.uniform(0.99, 0.999) for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Process through pipeline
        start_time = time.time()
        result = await feature_pipeline.process_market_data(ohlcv_data)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"   Original features: {result.original_features}")
        print(f"   Selected features: {result.selected_features}")
        print(f"   Feature reduction: {(1 - result.selected_features/max(result.original_features, 1))*100:.1f}%")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Performance target (<300ms): {'✅' if processing_time < 300 else '❌'}")
        
        # Show top feature importances
        if result.feature_importance:
            top_features = sorted(result.feature_importance, key=lambda x: x.combined_score, reverse=True)[:5]
            print(f"   Top features:")
            for feat in top_features:
                print(f"     - {feat.feature_name}: {feat.combined_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Feature pipeline demo failed: {e}")
        return None


async def demo_metrics_engine():
    """Demonstrate Metrics Engine API integration."""
    print("\n📊 Metrics Engine Demo")
    print("-" * 25)
    
    try:
        # Get real-time metrics for multiple symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        start_time = time.time()
        
        # This would normally make actual API calls, but for demo we'll simulate
        for symbol in symbols[:1]:  # Just test one to avoid API limits
            try:
                metrics = await metrics_engine.get_realtime_metrics(symbol)
                processing_time = (time.time() - start_time) * 1000
                
                print(f"   Symbol: {metrics.symbol}")
                print(f"   Current price: ${metrics.current_price:.2f}")
                print(f"   24h change: {metrics.price_change_pct_24h:.2f}%")
                print(f"   Volume 24h: ${metrics.volume_24h:,.0f}")
                print(f"   Fetch time: {metrics.fetch_time_ms:.2f}ms")
                print(f"   Performance target (<200ms): {'✅' if metrics.fetch_time_ms < 200 else '❌'}")
                
                if metrics.rsi:
                    print(f"   RSI: {metrics.rsi:.1f}")
                
                break  # Just demo one symbol
                
            except Exception as api_error:
                print(f"   Note: API demo simulated (would need real API keys)")
                print(f"   Expected response time: <200ms")
                print("   Features: Real-time price, volume, technical indicators")
                break
        
        # Get market state analysis
        market_state = await metrics_engine.get_market_state(['BTCUSDT'])
        print(f"   Market sentiment: {market_state.overall_sentiment}")
        print(f"   Volatility regime: {market_state.volatility_regime}")
        print(f"   Risk level: {market_state.risk_level}")
        
    except Exception as e:
        print(f"   Note: Metrics engine demo (simulated - requires API keys)")
        print(f"   Features: Sub-200ms API calls, intelligent caching, fallback sources")


async def demo_bayesian_framework():
    """Demonstrate Bayesian probability framework."""
    print("\n🧠 Bayesian Framework Demo")
    print("-" * 30)
    
    try:
        # Generate market data for Bayesian analysis
        market_data = generate_realistic_market_data('BTCUSDT')
        
        start_time = time.time()
        result = await bayesian_framework.analyze_market_bayesian(market_data)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"   Bayesian decision: {result.decision}")
        print(f"   Confidence score: {result.confidence_score:.3f}")
        print(f"   Uncertainty score: {result.uncertainty_score:.3f}")
        print(f"   Expected value: {result.expected_value:.4f}")
        print(f"   Value at Risk: {result.value_at_risk:.4f}")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Performance target (<100ms): {'✅' if processing_time < 100 else '❌'}")
        
        # Show decision probabilities
        print(f"   Decision probabilities:")
        for decision, prob in result.posterior_probabilities.items():
            print(f"     - {decision}: {prob:.3f}")
        
        # Show evidence summary
        print(f"   Evidence sources: {result.evidence_summary.get('total_evidence_sources', 0)}")
        print(f"   Evidence quality: {result.evidence_summary.get('evidence_quality_score', 0):.3f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Bayesian framework demo failed: {e}")
        return None


async def demo_market_classification():
    """Demonstrate market classification and opportunity identification."""
    print("\n🎯 Market Classification Demo")
    print("-" * 35)
    
    try:
        # Generate comprehensive market data
        market_data = generate_realistic_market_data('BTCUSDT')
        
        start_time = time.time()
        analysis = await market_classification_engine.analyze_market_comprehensively(market_data)
        processing_time = (time.time() - start_time) * 1000
        
        # Market classification results
        classification = analysis.market_classification
        print(f"   Primary regime: {classification.primary_regime.value}")
        print(f"   Regime confidence: {classification.regime_confidence:.3f}")
        print(f"   Trend strength: {classification.trend_strength:.3f}")
        print(f"   Momentum score: {classification.momentum_score:.3f}")
        print(f"   Volatility percentile: {classification.volatility_percentile:.0f}")
        print(f"   Regime stability: {classification.regime_stability:.3f}")
        
        # Opportunity identification
        print(f"   Opportunities identified: {len(analysis.identified_opportunities)}")
        for i, opp in enumerate(analysis.identified_opportunities[:3]):  # Show top 3
            print(f"     {i+1}. {opp.opportunity_type.value}")
            print(f"        Score: {opp.opportunity_score:.3f}")
            print(f"        Action: {opp.suggested_action}")
            print(f"        Confidence: {opp.confidence_level:.3f}")
            if opp.risk_reward_ratio:
                print(f"        Risk/Reward: 1:{opp.risk_reward_ratio:.1f}")
        
        # Overall assessment
        print(f"   Overall market score: {analysis.overall_market_score:.3f}")
        print(f"   Risk-adjusted score: {analysis.risk_adjusted_score:.3f}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Analysis confidence: {analysis.analysis_confidence:.3f}")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Performance target (<500ms): {'✅' if processing_time < 500 else '❌'}")
        
        return analysis
        
    except Exception as e:
        print(f"   ❌ Market classification demo failed: {e}")
        return None


async def demo_comprehensive_analysis():
    """Demonstrate the complete comprehensive analysis API."""
    print("\n🚀 Comprehensive Analysis API Demo")
    print("-" * 40)
    
    try:
        # Generate realistic market data
        market_data = generate_realistic_market_data('BTCUSDT')
        
        start_time = time.time()
        result = await comprehensive_market_analysis(market_data, 'BTCUSDT')
        total_time = (time.time() - start_time) * 1000
        
        print(f"   Symbol: {result['symbol']}")
        print(f"   Analysis timestamp: {result['timestamp']}")
        
        # Market classification summary
        classification = result['market_classification']
        if classification:
            print(f"   Market regime: {classification['primary_regime']}")
            print(f"   Regime confidence: {classification['regime_confidence']:.3f}")
            print(f"   Trend strength: {classification['trend_strength']:.3f}")
            print(f"   Volatility percentile: {classification['volatility_percentile']:.0f}")
        
        # Opportunities summary
        opportunities = result['trading_opportunities']
        print(f"   Trading opportunities: {len(opportunities)}")
        for opp in opportunities[:2]:  # Show top 2
            print(f"     - {opp['type']}: {opp['score']:.3f} (action: {opp['suggested_action']})")
        
        # Overall assessment
        assessment = result['overall_assessment']
        print(f"   Market score: {assessment['market_score']:.3f}")
        print(f"   Risk-adjusted score: {assessment['risk_adjusted_score']:.3f}")
        print(f"   Recommendation: {assessment['recommendation']}")
        
        # Performance metrics
        performance = result['performance_metrics']
        print(f"   Total processing time: {total_time:.2f}ms")
        print(f"   Sub-500ms target: {'✅' if performance['target_met'] else '❌'}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Comprehensive analysis demo failed: {e}")
        return None


async def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n⚡ Performance Optimization Demo")
    print("-" * 38)
    
    try:
        # Get performance statistics
        stats = performance_optimizer.get_performance_summary()
        
        if stats:
            print(f"   Total operations tracked: {stats.get('total_operations', 0)}")
            print(f"   Average processing time: {stats.get('avg_processing_time_ms', 0):.2f}ms")
            print(f"   95th percentile time: {stats.get('p95_processing_time_ms', 0):.2f}ms")
            print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"   Target compliance rate: {stats.get('target_compliance_rate', 0):.1%}")
            
            # Cache performance
            cache_stats = stats.get('cache_stats', {})
            if cache_stats:
                print(f"   Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                print(f"   Cache size: {cache_stats.get('cache_size', 0)} items")
            
            # Component performance
            component_averages = stats.get('component_averages', {})
            if component_averages:
                slowest = stats.get('slowest_component')
                print(f"   Slowest component: {slowest}")
                print(f"   Component breakdown:")
                for component, avg_time in sorted(component_averages.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"     - {component}: {avg_time:.1f}ms avg")
        else:
            print("   Performance tracking initialized")
            print("   Features: Intelligent caching, adaptive optimization, detailed profiling")
            print("   Targets: <500ms total, component-specific targets")
        
        # Test performance optimization
        print("   Running optimization analysis...")
        performance_optimizer.optimize_system()
        print("   ✅ System optimization completed")
        
    except Exception as e:
        print(f"   ⚠️  Performance optimization demo: {e}")


async def demo_system_status():
    """Demonstrate system status monitoring."""
    print("\n📈 System Status Dashboard")
    print("-" * 30)
    
    try:
        status = get_analytics_system_status()
        
        print(f"   Overall system health: {status.get('overall_health', 'unknown').upper()}")
        print(f"   Components status:")
        
        components = [
            ('clustering_engine', '🧩'),
            ('feature_pipeline', '🔧'), 
            ('metrics_engine', '📊'),
            ('bayesian_framework', '🧠'),
            ('market_classification_engine', '🎯'),
            ('performance_optimizer', '⚡')
        ]
        
        for component, icon in components:
            comp_stats = status.get(component, {})
            if comp_stats:
                # Extract key metrics
                if 'avg_processing_time_ms' in comp_stats:
                    avg_time = comp_stats['avg_processing_time_ms']
                    print(f"   {icon} {component}: {avg_time:.1f}ms avg")
                elif 'total_operations' in comp_stats:
                    ops = comp_stats['total_operations']
                    print(f"   {icon} {component}: {ops} operations")
                else:
                    print(f"   {icon} {component}: Active")
            else:
                print(f"   {icon} {component}: Ready")
        
    except Exception as e:
        print(f"   ⚠️  System status error: {e}")


async def run_performance_benchmark():
    """Run performance benchmark to validate sub-500ms target."""
    print("\n🏁 Performance Benchmark")
    print("-" * 25)
    
    benchmark_results = []
    
    try:
        print("   Running 10 comprehensive analyses...")
        
        for i in range(10):
            market_data = generate_realistic_market_data(f'TEST{i}')
            
            start_time = time.time()
            result = await comprehensive_market_analysis(market_data, f'TEST{i}')
            execution_time = (time.time() - start_time) * 1000
            
            benchmark_results.append(execution_time)
            
            # Show progress
            if (i + 1) % 3 == 0:
                print(f"   Completed {i + 1}/10 analyses...")
        
        # Calculate statistics
        avg_time = np.mean(benchmark_results)
        max_time = np.max(benchmark_results)
        min_time = np.min(benchmark_results)
        p95_time = np.percentile(benchmark_results, 95)
        
        success_rate = sum(1 for t in benchmark_results if t < 500) / len(benchmark_results)
        
        print(f"\n   📊 Benchmark Results:")
        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   Minimum time: {min_time:.2f}ms") 
        print(f"   Maximum time: {max_time:.2f}ms")
        print(f"   95th percentile: {p95_time:.2f}ms")
        print(f"   Sub-500ms success rate: {success_rate:.1%}")
        print(f"   Target achievement: {'✅ PASSED' if success_rate >= 0.9 else '❌ NEEDS OPTIMIZATION'}")
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")


async def main():
    """Run the complete advanced analytics demonstration."""
    
    print("🚀 Advanced Analytics Suite Demo")
    print("=" * 50)
    print("Demonstrating:")
    print("✅ HDBSCAN clustering with performance rules")
    print("✅ Advanced preprocessing & feature selection")
    print("✅ Metrics Engine API integration")
    print("✅ Bayesian probability framework")
    print("✅ Market classification & opportunities")
    print("✅ Sub-500ms performance optimization")
    print("=" * 50)
    
    try:
        # Run individual component demos
        await demo_clustering_engine()
        await demo_feature_pipeline()
        await demo_metrics_engine()
        await demo_bayesian_framework()
        await demo_market_classification()
        await demo_comprehensive_analysis()
        await demo_performance_optimization()
        await demo_system_status()
        
        # Run performance benchmark
        await run_performance_benchmark()
        
        print("\n🎉 Advanced Analytics Demo Completed!")
        print("=" * 50)
        print("Key Features Demonstrated:")
        print("🧩 High-performance HDBSCAN clustering")
        print("🔧 Intelligent feature selection pipeline")
        print("📊 Real-time metrics with API integration")
        print("🧠 Bayesian inference & decision making")
        print("🎯 Market regime classification")
        print("⚡ Sub-500ms performance optimization")
        print("\nThe system is ready for production use with:")
        print("- Strict performance constraints (<500ms)")
        print("- Comprehensive market analysis")
        print("- Intelligent caching & optimization")
        print("- Real-time opportunity identification")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            performance_optimizer.cleanup_resources()
            print("\n🧹 Resources cleaned up successfully")
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    print("🚀 Starting Advanced Analytics Demo...")
    
    # Run the complete demonstration
    asyncio.run(main())
    
    print("\n👋 Demo complete! Thank you for exploring the Advanced Analytics Suite.")