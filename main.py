#!/usr/bin/env python3
"""
Crypto Trend Analyzer AI Agent - Main Application Entry Point

A comprehensive cryptocurrency analysis system with:
- Real-time market data streaming
- AI-powered technical analysis
- Advanced signal generation
- Portfolio management
- Risk assessment
- Web dashboard
- Backtesting capabilities
"""

import asyncio
import argparse
import sys
import signal
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.config import Config
from src.utils.logging_config import crypto_logger, log_error
from src.utils.monitoring import start_monitoring, stop_monitoring, add_alert_callback
from src.data_sources.market_data import MarketDataFetcher
from src.data_sources.realtime_streams import RealTimeDataStreamer
from src.signals.signal_generator import AdvancedSignalGenerator
from src.portfolio.portfolio_manager import PortfolioManager
from src.dashboard.main import app

# Import enhanced core system components
from src.core import (
    control_center,
    black_box_processor,
    validation_engine,
    trading_control_system,
    error_handler,
    handle_error_async
)

class CryptoTrendAnalyzer:
    """Main application class for the Crypto Trend Analyzer."""
    
    def __init__(self):
        self.config = Config()
        self.market_fetcher = None
        self.realtime_streamer = None
        self.signal_generator = AdvancedSignalGenerator()
        self.portfolio_manager = PortfolioManager()
        self.running = False
        
    async def initialize(self):
        """Initialize all components."""
        try:
            crypto_logger.logger.info("🚀 Starting Enhanced Crypto Trend Analyzer AI Agent...")
            
            # Initialize core system components first
            crypto_logger.logger.info("🔧 Initializing core system components...")
            await control_center.initialize()
            await black_box_processor.initialize()
            crypto_logger.logger.info("✓ Core system components initialized")
            
            # Initialize market data fetcher
            self.market_fetcher = MarketDataFetcher()
            crypto_logger.logger.info("✓ Market data fetcher initialized")
            
            # Initialize real-time streamer
            self.realtime_streamer = RealTimeDataStreamer()
            
            # Set up callbacks for real-time data
            self.realtime_streamer.add_callback('price_update', self._handle_price_update)
            self.realtime_streamer.add_callback('signal_update', self._handle_signal_update)
            self.realtime_streamer.add_callback('anomaly_alert', self._handle_anomaly_alert)
            crypto_logger.logger.info("✓ Real-time data streamer initialized")
            
            # Initialize legacy monitoring (keeping compatibility)
            start_monitoring()
            add_alert_callback(self._handle_system_alert)
            crypto_logger.logger.info("✓ Legacy monitoring initialized")
            
            crypto_logger.logger.info("🎯 Enhanced Crypto Trend Analyzer fully initialized!")
            crypto_logger.logger.info("🛡️ Enhanced features active: Error Handling, Validation Models, Black Box Processing, Circuit Breakers")
            
        except Exception as e:
            await handle_error_async(e, {'component': 'initialization'})
            raise
    
    async def start_analysis(self):
        """Start the main analysis loop."""
        self.running = True
        crypto_logger.logger.info("📊 Starting market analysis...")
        
        try:
            # Start real-time streams
            if self.config.SYMBOLS:
                self.realtime_streamer.start_streaming(self.config.SYMBOLS)
                crypto_logger.logger.info(f"✓ Real-time streaming started for {len(self.config.SYMBOLS)} symbols")
            
            # Start main analysis loop
            await self._main_analysis_loop()
            
        except Exception as e:
            log_error(e, {'component': 'analysis_loop'})
            raise
    
    async def _main_analysis_loop(self):
        """Main analysis loop - generates signals and manages portfolio."""
        crypto_logger.logger.info("🔄 Main analysis loop started")
        
        while self.running:
            try:
                await self._analyze_and_generate_signals()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                log_error(e, {'component': 'main_analysis_loop'})
                await asyncio.sleep(60)
    
    async def _analyze_and_generate_signals(self):
        """Analyze market data and generate trading signals."""
        try:
            async with self.market_fetcher as fetcher:
                for i, crypto_id in enumerate(self.config.CRYPTOS[:5]):  # Analyze top 5
                    symbol = self.config.SYMBOLS[i] if i < len(self.config.SYMBOLS) else 'BTCUSDT'
                    
                    # Get comprehensive market data
                    market_data = await fetcher.get_comprehensive_data(crypto_id, symbol)
                    
                    if not market_data or not market_data.get('data_quality', {}).get('score', 0):
                        crypto_logger.logger.warning(f"Insufficient data for {crypto_id}")
                        continue
                    
                    # Generate trading signals
                    signal_data = self.signal_generator.generate_comprehensive_signals(market_data)
                    
                    if signal_data:
                        # Log signal generation
                        crypto_logger.log_signal(signal_data)
                        
                        # Process signal through enhanced black box system
                        await self._handle_enhanced_signal_processing(signal_data)
                        
                        crypto_logger.logger.info(
                            f"✓ Signal generated for {symbol}: "
                            f"{signal_data['final_signal']['signal']} "
                            f"(confidence: {signal_data['final_signal']['confidence_score']:.1f}%)"
                        )
                    
                    # Small delay between symbols
                    await asyncio.sleep(2)
                    
        except Exception as e:
            log_error(e, {'component': 'signal_generation'})
    
    async def _handle_enhanced_signal_processing(self, signal_data):
        """Enhanced signal processing using black box system with validation and controls."""
        try:
            symbol = signal_data.get('symbol', 'Unknown')
            
            # Get current portfolio data
            portfolio_metrics = self.portfolio_manager.get_portfolio_metrics()
            portfolio_data = {
                'total_value': getattr(portfolio_metrics, 'total_value', 100000),
                'cash_balance': getattr(portfolio_metrics, 'cash_balance', 50000),
                'positions': getattr(self.portfolio_manager, 'positions', {})
            }
            
            # Process through black box system
            crypto_logger.logger.info(f"🤖 Processing signal for {symbol} through Black Box system...")
            black_box_result = await black_box_processor.process_trading_signal(signal_data, portfolio_data)
            
            # Check trading controls before execution
            trade_data = {
                'type': black_box_result.decision.value,
                'symbol': symbol,
                'size': 1000  # Would be calculated based on decision
            }
            
            can_execute, control_reason = await trading_control_system.can_execute_trade(trade_data)
            
            if not can_execute:
                crypto_logger.logger.warning(f"🚫 Trade blocked by controls: {control_reason}")
                return
            
            # Execute trade based on black box decision
            await self._execute_black_box_decision(black_box_result, signal_data, portfolio_data)
            
        except Exception as e:
            await handle_error_async(e, {'component': 'enhanced_signal_processing', 'symbol': signal_data.get('symbol')})
    
    async def _execute_black_box_decision(self, black_box_result, signal_data, portfolio_data):
        """Execute trading decision from black box processing."""
        try:
            symbol = signal_data.get('symbol')
            decision = black_box_result.decision
            confidence = black_box_result.confidence
            
            crypto_logger.logger.info(
                f"🎯 Black Box Decision: {decision.value} for {symbol} "
                f"(confidence: {confidence.value}, expected return: {black_box_result.expected_return:.3f})"
            )
            
            # Log supporting and conflicting evidence
            if black_box_result.supporting_evidence:
                crypto_logger.logger.info(f"📊 Supporting: {', '.join(black_box_result.supporting_evidence[:2])}")
            if black_box_result.conflicting_evidence:
                crypto_logger.logger.info(f"⚠️ Conflicting: {', '.join(black_box_result.conflicting_evidence[:2])}")
            
            # Only execute if confidence is sufficient and decision is actionable
            if decision.value == 'no_trade':
                crypto_logger.logger.info(f"⏸️ No trade executed for {symbol}")
                return
            
            if confidence.value in ['very_low', 'low']:
                crypto_logger.logger.warning(f"⚠️ Low confidence trade skipped for {symbol}")
                return
            
            # Get trade parameters from original signal
            final_signal = signal_data.get('final_signal', {})
            entry_price = final_signal.get('entry_price')
            
            if not entry_price:
                crypto_logger.logger.warning(f"❌ No entry price available for {symbol}")
                return
            
            # Execute based on decision type
            if decision.value in ['strong_buy', 'buy']:
                await self._execute_buy_order(symbol, entry_price, final_signal, portfolio_data, black_box_result)
            elif decision.value in ['strong_sell', 'sell']:
                await self._execute_sell_order(symbol, entry_price, final_signal, portfolio_data)
            elif decision.value == 'hold':
                crypto_logger.logger.info(f"🤝 Holding position for {symbol}")
            
        except Exception as e:
            await handle_error_async(e, {'component': 'black_box_execution', 'symbol': signal_data.get('symbol')})
    
    async def _execute_buy_order(self, symbol, entry_price, final_signal, portfolio_data, black_box_result):
        """Execute buy order with enhanced position sizing."""
        try:
            # Calculate position size based on black box recommendations and risk
            base_position_ratio = 0.1  # 10% of portfolio
            confidence_multiplier = {
                'very_high': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4,
                'very_low': 0.2
            }.get(black_box_result.confidence.value, 0.5)
            
            # Adjust for expected return and risk
            return_multiplier = min(1.5, max(0.5, 1 + black_box_result.expected_return))
            risk_multiplier = max(0.3, 1 - black_box_result.risk_adjusted_return)
            
            position_ratio = base_position_ratio * confidence_multiplier * return_multiplier * risk_multiplier
            
            max_position_value = portfolio_data['total_value'] * position_ratio
            quantity = min(
                max_position_value / entry_price,
                portfolio_data['cash_balance'] / entry_price * 0.8
            )
            
            if quantity > 0 and portfolio_data['cash_balance'] > quantity * entry_price:
                success = self.portfolio_manager.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=final_signal.get('stop_loss'),
                    take_profit=final_signal.get('take_profit')
                )
                
                if success:
                    crypto_logger.logger.info(
                        f"📈 Enhanced buy executed: {symbol} - {quantity:.4f} @ ${entry_price} "
                        f"(size ratio: {position_ratio:.2%})"
                    )
                else:
                    crypto_logger.logger.warning(f"❌ Failed to execute buy order: {symbol}")
            else:
                crypto_logger.logger.warning(f"💰 Insufficient funds for {symbol} buy order")
                
        except Exception as e:
            await handle_error_async(e, {'component': 'buy_execution', 'symbol': symbol})
    
    async def _execute_sell_order(self, symbol, entry_price, final_signal, portfolio_data):
        """Execute sell order."""
        try:
            # Close existing position if any
            if symbol in portfolio_data.get('positions', {}):
                success = self.portfolio_manager.close_position(symbol, entry_price)
                if success:
                    crypto_logger.logger.info(f"📉 Enhanced sell executed: {symbol} @ ${entry_price}")
                else:
                    crypto_logger.logger.warning(f"❌ Failed to execute sell order: {symbol}")
            else:
                crypto_logger.logger.info(f"ℹ️ No position to sell for {symbol}")
                
        except Exception as e:
            await handle_error_async(e, {'component': 'sell_execution', 'symbol': symbol})
    
    async def _handle_generated_signal(self, signal_data):
        """Legacy signal handler - kept for backward compatibility."""
        # This method is kept for backward compatibility but enhanced processing is preferred
        await self._handle_enhanced_signal_processing(signal_data)
    
    def _handle_price_update(self, price_data):
        """Handle real-time price updates."""
        try:
            # Update portfolio prices
            symbol = price_data.get('symbol')
            price = price_data.get('price')
            
            if symbol and price:
                self.portfolio_manager.update_prices({symbol: price})
                
        except Exception as e:
            log_error(e, {'component': 'price_update_handler'})
    
    def _handle_signal_update(self, signal_data):
        """Handle signal updates from real-time analysis."""
        try:
            crypto_logger.logger.debug(f"Real-time signal update: {signal_data}")
            
        except Exception as e:
            log_error(e, {'component': 'signal_update_handler'})
    
    def _handle_anomaly_alert(self, anomaly_data):
        """Handle market anomaly alerts."""
        try:
            crypto_logger.log_anomaly(anomaly_data)
            
        except Exception as e:
            log_error(e, {'component': 'anomaly_alert_handler'})
    
    def _handle_system_alert(self, alert):
        """Handle system monitoring alerts."""
        try:
            crypto_logger.logger.warning(f"System alert: {alert.title} - {alert.message}")
            
        except Exception as e:
            log_error(e, {'component': 'system_alert_handler'})
    
    def stop(self):
        """Stop the analyzer gracefully."""
        crypto_logger.logger.info("🛑 Stopping Enhanced Crypto Trend Analyzer...")
        
        self.running = False
        
        # Stop real-time streaming
        if self.realtime_streamer:
            self.realtime_streamer.stop_streaming()
        
        # Stop enhanced core components
        try:
            control_center.stop_monitoring()
            trading_control_system.stop()
            crypto_logger.logger.info("✓ Enhanced core components stopped")
        except Exception as e:
            crypto_logger.logger.error(f"Error stopping core components: {e}")
        
        # Stop legacy monitoring
        stop_monitoring()
        
        crypto_logger.logger.info("✓ Enhanced Crypto Trend Analyzer stopped")

async def run_analyzer():
    """Run the crypto trend analyzer."""
    analyzer = CryptoTrendAnalyzer()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        analyzer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await analyzer.initialize()
        await analyzer.start_analysis()
        
    except KeyboardInterrupt:
        analyzer.stop()
    except Exception as e:
        log_error(e, {'component': 'main_application'})
        analyzer.stop()
        raise

def run_dashboard():
    """Run the web dashboard."""
    import uvicorn
    
    crypto_logger.logger.info("🌐 Starting web dashboard...")
    
    try:
        uvicorn.run(
            "src.dashboard.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except Exception as e:
        log_error(e, {'component': 'dashboard'})
        raise

def run_backtest():
    """Run backtesting on historical data."""
    from src.backtesting.backtest_engine import BacktestEngine
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    
    crypto_logger.logger.info("📊 Running backtest...")
    
    try:
        backtest_engine = BacktestEngine()
        
        # Generate sample historical data for demo
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now() - timedelta(days=1)
        
        # Create sample OHLCV data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        base_price = 45000
        prices = []
        volumes = []
        
        for _ in range(len(dates)):
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
        results = backtest_engine.run_backtest('BTCUSDT', data, start_date, end_date)
        
        # Generate and print report
        report = backtest_engine.generate_backtest_report(results)
        print(report)
        
        crypto_logger.log_backtest_result({
            'symbol': 'BTCUSDT',
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate
        })
        
        crypto_logger.logger.info("✓ Backtest completed successfully")
        
    except Exception as e:
        log_error(e, {'component': 'backtesting'})
        raise

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Crypto Trend Analyzer AI Agent')
    parser.add_argument('command', choices=['analyze', 'dashboard', 'backtest', 'all'], 
                       help='Command to run')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        crypto_logger.logger.setLevel(crypto_logger.logger.DEBUG)
    
    print("🚀 Crypto Trend Analyzer AI Agent")
    print("=" * 50)
    print(f"Command: {args.command}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        if args.command == 'analyze':
            asyncio.run(run_analyzer())
            
        elif args.command == 'dashboard':
            run_dashboard()
            
        elif args.command == 'backtest':
            run_backtest()
            
        elif args.command == 'all':
            # Run analyzer and dashboard concurrently
            async def run_all():
                analyzer_task = asyncio.create_task(run_analyzer())
                dashboard_task = asyncio.create_task(
                    asyncio.to_thread(run_dashboard)
                )
                
                await asyncio.gather(analyzer_task, dashboard_task)
            
            asyncio.run(run_all())
        
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        log_error(e, {'component': 'main'})
        sys.exit(1)

if __name__ == "__main__":
    main()