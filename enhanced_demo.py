#!/usr/bin/env python3
"""
Enhanced Crypto Trend Analyzer Demo

This script demonstrates the new enhanced algorithmic trading system architecture
including error handling, validation models, black box processing, and circuit breakers.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core import (
    control_center,
    black_box_processor,
    validation_engine,
    trading_control_system,
    error_handler
)
from src.utils.logging_config import crypto_logger


async def demo_enhanced_features():
    """Demonstrate the enhanced features."""
    
    print("\n🚀 Enhanced Crypto Trend Analyzer Demo")
    print("=" * 50)
    
    try:
        # Initialize core components
        print("\n🔧 Initializing Enhanced Core Components...")
        await control_center.initialize()
        await black_box_processor.initialize()
        
        print("✅ Core components initialized successfully!")
        
        # Demo 1: System Status Dashboard
        print("\n📊 System Status Dashboard:")
        dashboard = control_center.get_system_dashboard()
        print(f"   System Health: {dashboard['system_status']}")
        print(f"   Active Alerts: {dashboard['active_alerts']}")
        print(f"   Trading Status: {dashboard['trading_control_status']['trading_window_status']}")
        
        # Demo 2: Black Box Processing
        print("\n🤖 Black Box Processing Demo:")
        sample_signal = {
            'symbol': 'BTCUSDT',
            'final_signal': {
                'signal': 'BUY',
                'confidence_score': 75.5,
                'entry_price': 45000,
                'stop_loss': 44000,
                'take_profit': 46500
            },
            'technical_analysis': {
                'rsi': {'signal': 'BUY', 'value': 35},
                'macd': {'signal': 'BUY', 'value': 0.5},
                'bollinger': {'signal': 'NEUTRAL', 'value': 0}
            },
            'market_data': {
                'current_price': 45000,
                'volume_24h': 2000000000,
                'volatility': 0.03
            }
        }
        
        sample_portfolio = {
            'total_value': 100000,
            'cash_balance': 50000,
            'positions': {}
        }
        
        result = await black_box_processor.process_trading_signal(sample_signal, sample_portfolio)
        print(f"   Decision: {result.decision.value}")
        print(f"   Confidence: {result.confidence.value}")
        print(f"   Expected Return: {result.expected_return:.3f}")
        print(f"   Supporting Evidence: {result.supporting_evidence[:1]}")
        
        # Demo 3: Validation Models
        print("\n🛡️ Validation Models Demo:")
        validation_reports = await validation_engine.comprehensive_validation(sample_signal, sample_portfolio)
        
        for report in validation_reports:
            print(f"   {report.category.value}: {report.result.value} (confidence: {report.confidence_score:.2f})")
            if report.reasons:
                print(f"     Reasons: {', '.join(report.reasons[:1])}")
        
        # Demo 4: Trading Controls
        print("\n⚖️ Trading Controls Demo:")
        trade_data = {'type': 'buy', 'symbol': 'BTCUSDT', 'size': 1000}
        can_trade, reason = await trading_control_system.can_execute_trade(trade_data)
        print(f"   Can Execute Trade: {can_trade}")
        if not can_trade:
            print(f"   Reason: {reason}")
        
        # Demo 5: Circuit Breaker Status
        print("\n🔴 Circuit Breaker Status:")
        cb_status = trading_control_system.circuit_breaker_manager.get_circuit_breaker_status()
        for breaker_name, status in cb_status.items():
            print(f"   {breaker_name}: {status['state']} (triggers: {status['trigger_count']})")
        
        # Demo 6: Error Statistics
        print("\n📈 Error Statistics:")
        error_stats = error_handler.get_error_statistics()
        if error_stats:
            print(f"   Total Errors (last hour): {error_stats.get('total_errors_last_hour', 0)}")
            print(f"   Resolved Errors: {error_stats.get('resolved_errors', 0)}")
            if error_stats.get('errors_by_category'):
                print(f"   By Category: {error_stats['errors_by_category']}")
        else:
            print("   No errors recorded (system is healthy)")
        
        print("\n✅ Demo completed successfully!")
        print("\n🎯 Enhanced Features Summary:")
        print("   ✓ Error Handling Engine - Centralized error management")
        print("   ✓ Validation Models - Independent signal validation") 
        print("   ✓ Black Box Processing - ML + Probabilistic decision engine")
        print("   ✓ Trading Controls - Time windows and circuit breakers")
        print("   ✓ Monitoring Center - Real-time system oversight")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        try:
            await control_center.stop_monitoring()
            trading_control_system.stop()
            print("\n🛑 Demo components shut down gracefully")
        except Exception as e:
            print(f"⚠️ Shutdown warning: {e}")


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    
    print("\n🔧 Error Handling Demo:")
    print("-" * 30)
    
    try:
        # Simulate an error
        raise ValueError("Simulated trading error for demo")
    except Exception as e:
        # Use enhanced error handling
        error_event = await error_handler.handle_error(e, {
            'component': 'demo_trading_system',
            'symbol': 'BTCUSDT',
            'operation': 'signal_processing'
        })
        
        print(f"   Error ID: {error_event.id}")
        print(f"   Category: {error_event.category.value}")
        print(f"   Severity: {error_event.severity.value}")
        print(f"   Resolved: {error_event.resolved}")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Crypto Trend Analyzer Demo...")
    
    # Run the demo
    asyncio.run(demo_enhanced_features())
    
    # Run error handling demo
    asyncio.run(demo_error_handling())
    
    print("\n👋 Demo complete. Thank you for exploring the enhanced features!")