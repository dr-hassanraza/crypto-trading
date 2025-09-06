#!/usr/bin/env python3
"""
Advanced Crypto Trend Analyzer - Master Integration Module

This module combines all advanced features:
- ML-based price prediction
- Multi-exchange arbitrage detection  
- DeFi protocol analysis
- Social sentiment tracking
- Advanced risk modeling
- Real-time market intelligence
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from src.ml_models.price_prediction import ml_predictor
from src.arbitrage.arbitrage_detector import arbitrage_detector
from src.defi.defi_analyzer import defi_analyzer
from src.sentiment.social_sentiment import social_sentiment_analyzer
from src.signals.signal_generator import AdvancedSignalGenerator
from src.portfolio.portfolio_manager import PortfolioManager
from src.data_sources.market_data import MarketDataFetcher
from src.utils.logging_config import crypto_logger
from config.config import Config

class MasterCryptoAnalyzer:
    """Master analyzer combining all advanced crypto analysis capabilities."""
    
    def __init__(self):
        self.config = Config()
        
        # Initialize all components
        self.ml_predictor = ml_predictor
        self.arbitrage_detector = arbitrage_detector
        self.defi_analyzer = defi_analyzer
        self.sentiment_analyzer = social_sentiment_analyzer
        self.signal_generator = AdvancedSignalGenerator()
        self.portfolio_manager = PortfolioManager()
        self.market_fetcher = None
        
        # Analysis cache
        self.analysis_cache = {
            'predictions': {},
            'arbitrage': {},
            'defi': {},
            'sentiment': {},
            'signals': {},
            'last_updated': {}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'predictions_accuracy': {},
            'arbitrage_profits': {},
            'defi_yields': {},
            'signal_performance': {}
        }
        
    async def initialize(self):
        """Initialize all analysis components."""
        crypto_logger.logger.info("🚀 Initializing Master Crypto Analyzer...")
        
        try:
            # Initialize market data fetcher
            self.market_fetcher = MarketDataFetcher()
            
            # Initialize arbitrage detector
            await self.arbitrage_detector.initialize_exchanges()
            
            # Initialize DeFi analyzer
            await self.defi_analyzer.initialize_web3_connections()
            
            crypto_logger.logger.info("✅ Master Crypto Analyzer initialized successfully")
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Initialization error: {e}")
            raise
    
    async def run_comprehensive_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive analysis across all modules."""
        crypto_logger.logger.info(f"🔍 Running comprehensive analysis for {len(symbols)} symbols")
        
        analysis_results = {
            'overview': {
                'symbols_analyzed': symbols,
                'analysis_timestamp': datetime.now().isoformat(),
                'modules_used': [
                    'ml_prediction', 'arbitrage_detection', 'defi_analysis',
                    'sentiment_analysis', 'technical_signals', 'portfolio_optimization'
                ]
            },
            'results': {}
        }
        
        # Run all analyses concurrently for efficiency
        tasks = [
            self._run_ml_predictions(symbols),
            self._run_arbitrage_analysis(symbols),
            self._run_defi_analysis(),
            self._run_sentiment_analysis(symbols),
            self._run_signal_generation(symbols),
            self._run_portfolio_analysis()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analysis_results['results'] = {
            'ml_predictions': results[0] if not isinstance(results[0], Exception) else {},
            'arbitrage_opportunities': results[1] if not isinstance(results[1], Exception) else {},
            'defi_opportunities': results[2] if not isinstance(results[2], Exception) else {},
            'sentiment_analysis': results[3] if not isinstance(results[3], Exception) else {},
            'trading_signals': results[4] if not isinstance(results[4], Exception) else {},
            'portfolio_insights': results[5] if not isinstance(results[5], Exception) else {}
        }
        
        # Generate master recommendations
        analysis_results['recommendations'] = await self._generate_master_recommendations(
            analysis_results['results']
        )
        
        # Update cache
        self.analysis_cache.update({
            'last_comprehensive_analysis': analysis_results,
            'last_updated': datetime.now()
        })
        
        crypto_logger.logger.info("✅ Comprehensive analysis completed")
        return analysis_results
    
    async def _run_ml_predictions(self, symbols: List[str]) -> Dict[str, Any]:
        """Run ML-based price predictions."""
        crypto_logger.logger.info("🤖 Running ML price predictions")
        
        predictions = {}
        
        try:
            async with self.market_fetcher as fetcher:
                for symbol in symbols:
                    crypto_id = symbol.lower().replace('usdt', '').replace('usd', '')
                    
                    # Get historical data for ML training
                    comprehensive_data = await fetcher.get_comprehensive_data(crypto_id, symbol)
                    
                    if comprehensive_data and comprehensive_data.get('ohlcv_data'):
                        # Get daily OHLCV data
                        ohlcv_data = comprehensive_data['ohlcv_data'].get('1d')
                        
                        if ohlcv_data is not None and not ohlcv_data.empty:
                            # Train models if not already trained
                            if symbol not in self.ml_predictor.models:
                                trained_models = self.ml_predictor.train_models(ohlcv_data, symbol)
                                
                                if trained_models:
                                    crypto_logger.logger.info(f"✓ ML models trained for {symbol}")
                            
                            # Generate predictions
                            symbol_predictions = self.ml_predictor.predict_prices(symbol, ohlcv_data)
                            
                            if symbol_predictions:
                                predictions[symbol] = {
                                    'predictions': symbol_predictions,
                                    'model_performance': self.ml_predictor.get_model_performance_report(symbol),
                                    'feature_importance': self.ml_predictor.get_feature_importance(symbol),
                                    'data_quality': comprehensive_data.get('data_quality', {})
                                }
                                
                                crypto_logger.logger.info(f"✓ Predictions generated for {symbol}")
            
            self.analysis_cache['predictions'] = predictions
            return predictions
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ ML predictions error: {e}")
            return {}
    
    async def _run_arbitrage_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run arbitrage opportunity detection."""
        crypto_logger.logger.info("📊 Analyzing arbitrage opportunities")
        
        try:
            # Fetch prices from all exchanges
            all_prices = await self.arbitrage_detector.fetch_all_prices(symbols)
            
            if not all_prices:
                return {}
            
            # Detect simple arbitrage
            simple_opportunities = self.arbitrage_detector.detect_simple_arbitrage(all_prices)
            
            # Detect triangular arbitrage
            triangular_opportunities = self.arbitrage_detector.detect_triangular_arbitrage(all_prices)
            
            arbitrage_results = {
                'simple_arbitrage': [
                    {
                        'symbol': opp.symbol,
                        'buy_exchange': opp.buy_exchange,
                        'sell_exchange': opp.sell_exchange,
                        'buy_price': opp.buy_price,
                        'sell_price': opp.sell_price,
                        'spread_pct': opp.spread_pct,
                        'potential_profit': opp.potential_profit,
                        'risk_score': opp.risk_score,
                        'confidence': opp.confidence,
                        'execution_time': opp.execution_time_estimate
                    } for opp in simple_opportunities
                ],
                'triangular_arbitrage': [
                    {
                        'symbol_a': opp.symbol_a,
                        'symbol_b': opp.symbol_b,
                        'base_currency': opp.base_currency,
                        'exchange': opp.exchange,
                        'profit_pct': opp.profit_pct,
                        'execution_path': opp.execution_path,
                        'volume_constraint': opp.volume_constraint
                    } for opp in triangular_opportunities
                ],
                'summary': {
                    'total_simple_opportunities': len(simple_opportunities),
                    'total_triangular_opportunities': len(triangular_opportunities),
                    'best_simple_spread': max([opp.spread_pct for opp in simple_opportunities]) if simple_opportunities else 0,
                    'best_triangular_profit': max([opp.profit_pct for opp in triangular_opportunities]) if triangular_opportunities else 0,
                    'exchanges_monitored': list(all_prices.keys())
                }
            }
            
            self.analysis_cache['arbitrage'] = arbitrage_results
            return arbitrage_results
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Arbitrage analysis error: {e}")
            return {}
    
    async def _run_defi_analysis(self) -> Dict[str, Any]:
        """Run DeFi protocol and yield farming analysis."""
        crypto_logger.logger.info("🏦 Analyzing DeFi opportunities")
        
        try:
            # Fetch yield opportunities
            yield_opportunities = await self.defi_analyzer.fetch_yield_opportunities()
            
            # Calculate optimization for different risk profiles
            conservative_strategy = self.defi_analyzer.calculate_yield_optimization(10000, 'conservative')
            moderate_strategy = self.defi_analyzer.calculate_yield_optimization(10000, 'moderate')
            aggressive_strategy = self.defi_analyzer.calculate_yield_optimization(10000, 'aggressive')
            
            defi_results = {
                'yield_opportunities': yield_opportunities,
                'strategy_recommendations': {
                    'conservative': conservative_strategy,
                    'moderate': moderate_strategy,
                    'aggressive': aggressive_strategy
                },
                'top_yields': {
                    category: sorted(opps, key=lambda x: x.yield_pct, reverse=True)[:5]
                    for category, opps in yield_opportunities.items()
                },
                'risk_analysis': {
                    'avg_apy_by_risk': self._calculate_avg_apy_by_risk(yield_opportunities),
                    'protocol_distribution': self._analyze_protocol_distribution(yield_opportunities)
                }
            }
            
            self.analysis_cache['defi'] = defi_results
            return defi_results
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ DeFi analysis error: {e}")
            return {}
    
    async def _run_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run social sentiment analysis."""
        crypto_logger.logger.info("💭 Analyzing social sentiment")
        
        try:
            # Analyze sentiment for all symbols
            sentiment_data = await self.sentiment_analyzer.analyze_crypto_sentiment(symbols, 24)
            
            # Get trending topics
            trending_topics = await self.sentiment_analyzer.detect_trending_topics(4)
            
            # Track influencer signals
            influencer_signals = await self.sentiment_analyzer.track_influencer_signals()
            
            # Generate sentiment report
            sentiment_report = self.sentiment_analyzer.generate_sentiment_report(sentiment_data)
            
            sentiment_results = {
                'sentiment_data': {
                    symbol: {
                        'sentiment_score': data.sentiment_score,
                        'volume': data.volume,
                        'engagement': data.engagement,
                        'trending_score': data.trending_score,
                        'fear_greed_indicator': data.fear_greed_indicator,
                        'sample_posts': data.sample_posts[:3]
                    } for symbol, data in sentiment_data.items()
                },
                'trending_topics': [
                    {
                        'keyword': topic.keyword,
                        'platform': topic.platform,
                        'mention_count': topic.mention_count,
                        'growth_rate': topic.growth_rate,
                        'sentiment_score': topic.sentiment_score
                    } for topic in trending_topics
                ],
                'influencer_signals': [
                    {
                        'username': signal.username,
                        'platform': signal.platform,
                        'sentiment_score': signal.sentiment_score,
                        'influence_score': signal.influence_score,
                        'mentioned_cryptos': signal.mentioned_cryptos,
                        'post_content': signal.post_content[:150]
                    } for signal in influencer_signals
                ],
                'sentiment_report': sentiment_report
            }
            
            self.analysis_cache['sentiment'] = sentiment_results
            return sentiment_results
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Sentiment analysis error: {e}")
            return {}
    
    async def _run_signal_generation(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate advanced trading signals."""
        crypto_logger.logger.info("📈 Generating trading signals")
        
        signals = {}
        
        try:
            async with self.market_fetcher as fetcher:
                for symbol in symbols:
                    crypto_id = symbol.lower().replace('usdt', '').replace('usd', '')
                    
                    # Get comprehensive market data
                    comprehensive_data = await fetcher.get_comprehensive_data(crypto_id, symbol)
                    
                    if comprehensive_data:
                        # Generate signals
                        signal_data = self.signal_generator.generate_comprehensive_signals(comprehensive_data)
                        
                        if signal_data:
                            signals[symbol] = {
                                'final_signal': signal_data['final_signal'],
                                'signal_strength': signal_data['signal_strength'],
                                'confidence_score': signal_data['confidence_score'],
                                'component_signals': signal_data['component_signals'],
                                'data_quality': signal_data['data_quality']
                            }
            
            self.analysis_cache['signals'] = signals
            return signals
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Signal generation error: {e}")
            return {}
    
    async def _run_portfolio_analysis(self) -> Dict[str, Any]:
        """Run portfolio analysis and optimization."""
        crypto_logger.logger.info("💼 Analyzing portfolio")
        
        try:
            # Get current portfolio metrics
            portfolio_metrics = self.portfolio_manager.get_portfolio_metrics()
            
            # Get risk assessment
            risk_assessment = self.portfolio_manager.get_risk_assessment()
            
            # Get optimization suggestions
            optimization = self.portfolio_manager.optimize_portfolio()
            
            # Generate performance report
            performance_report = self.portfolio_manager.get_performance_report()
            
            portfolio_results = {
                'current_metrics': {
                    'total_value': portfolio_metrics.total_value,
                    'cash_balance': portfolio_metrics.cash_balance,
                    'total_pnl': portfolio_metrics.total_pnl,
                    'total_pnl_pct': portfolio_metrics.total_pnl_pct,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'max_drawdown': portfolio_metrics.max_drawdown,
                    'positions_count': portfolio_metrics.positions_count
                },
                'risk_assessment': risk_assessment,
                'optimization': optimization,
                'performance_report': performance_report
            }
            
            return portfolio_results
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Portfolio analysis error: {e}")
            return {}
    
    async def _generate_master_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate master recommendations based on all analysis results."""
        crypto_logger.logger.info("🎯 Generating master recommendations")
        
        recommendations = {
            'immediate_actions': [],
            'short_term_strategy': [],
            'long_term_strategy': [],
            'risk_warnings': [],
            'opportunity_alerts': [],
            'market_outlook': {}
        }
        
        try:
            # Process ML predictions
            predictions = analysis_results.get('ml_predictions', {})
            for symbol, pred_data in predictions.items():
                pred_1h = pred_data.get('predictions', {}).get('1h', {}).get('ensemble', {})
                if pred_1h and pred_1h.get('price', 0) > 0:
                    current_price = pred_data['predictions']['metadata']['current_price']
                    predicted_price = pred_1h['price']
                    change_pct = ((predicted_price - current_price) / current_price) * 100
                    
                    if abs(change_pct) > 5 and pred_1h.get('confidence', 0) > 70:
                        direction = "increase" if change_pct > 0 else "decrease"
                        recommendations['immediate_actions'].append(
                            f"{symbol}: ML model predicts {change_pct:.1f}% {direction} in next hour "
                            f"(confidence: {pred_1h['confidence']:.1f}%)"
                        )
            
            # Process arbitrage opportunities
            arbitrage = analysis_results.get('arbitrage_opportunities', {})
            simple_arb = arbitrage.get('simple_arbitrage', [])
            for opp in simple_arb[:3]:  # Top 3
                if opp.get('spread_pct', 0) > 2 and opp.get('confidence', 0) > 60:
                    recommendations['immediate_actions'].append(
                        f"Arbitrage: {opp['symbol']} - {opp['spread_pct']:.1f}% spread between "
                        f"{opp['buy_exchange']} and {opp['sell_exchange']}"
                    )
            
            # Process DeFi opportunities
            defi = analysis_results.get('defi_opportunities', {})
            for category, opportunities in defi.get('yield_opportunities', {}).items():
                top_yield = max(opportunities, key=lambda x: x.yield_pct) if opportunities else None
                if top_yield and top_yield.yield_pct > 15:
                    recommendations['long_term_strategy'].append(
                        f"High yield {category}: {top_yield.protocol} offering "
                        f"{top_yield.yield_pct:.1f}% APY on {top_yield.asset}"
                    )
            
            # Process sentiment analysis
            sentiment = analysis_results.get('sentiment_analysis', {})
            sentiment_report = sentiment.get('sentiment_report', {})
            overall_sentiment = sentiment_report.get('overall_market_sentiment', {})
            
            if overall_sentiment:
                sentiment_score = overall_sentiment.get('score', 0)
                sentiment_class = overall_sentiment.get('classification', 'NEUTRAL')
                
                if abs(sentiment_score) > 0.3:
                    recommendations['market_outlook']['sentiment'] = (
                        f"Market sentiment is {sentiment_class} (score: {sentiment_score:.2f}). "
                        f"Social media shows {sentiment_class.lower().replace('_', ' ')} bias."
                    )
            
            # Process trading signals
            signals = analysis_results.get('trading_signals', {})
            high_confidence_signals = []
            
            for symbol, signal_data in signals.items():
                final_signal = signal_data.get('final_signal', {})
                if (final_signal.get('confidence_score', 0) > 75 and 
                    final_signal.get('signal') in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']):
                    high_confidence_signals.append({
                        'symbol': symbol,
                        'signal': final_signal['signal'],
                        'confidence': final_signal['confidence_score'],
                        'entry_price': final_signal.get('entry_price'),
                        'stop_loss': final_signal.get('stop_loss'),
                        'take_profit': final_signal.get('take_profit')
                    })
            
            for signal in high_confidence_signals:
                recommendations['short_term_strategy'].append(
                    f"{signal['symbol']}: {signal['signal']} signal with {signal['confidence']:.1f}% confidence"
                )
            
            # Add risk warnings
            portfolio_risk = analysis_results.get('portfolio_insights', {}).get('risk_assessment', {})
            if portfolio_risk.get('risk_score', 0) > 70:
                recommendations['risk_warnings'].append(
                    f"Portfolio risk score is high ({portfolio_risk['risk_score']}). "
                    "Consider reducing position sizes or adding hedges."
                )
            
            # Market outlook summary
            recommendations['market_outlook']['summary'] = self._generate_market_outlook_summary(analysis_results)
            
        except Exception as e:
            crypto_logger.logger.error(f"❌ Error generating recommendations: {e}")
        
        return recommendations
    
    def _generate_market_outlook_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate overall market outlook summary."""
        try:
            # Analyze different components for overall outlook
            outlook_factors = []
            
            # ML predictions outlook
            predictions = analysis_results.get('ml_predictions', {})
            if predictions:
                avg_1h_change = np.mean([
                    ((pred['predictions'].get('1h', {}).get('ensemble', {}).get('price', pred['predictions']['metadata']['current_price']) 
                      - pred['predictions']['metadata']['current_price']) 
                     / pred['predictions']['metadata']['current_price'] * 100)
                    for pred in predictions.values()
                    if pred.get('predictions', {}).get('1h', {}).get('ensemble', {}).get('price')
                ])
                
                if abs(avg_1h_change) > 2:
                    direction = "bullish" if avg_1h_change > 0 else "bearish"
                    outlook_factors.append(f"ML models show {direction} short-term bias")
            
            # Sentiment outlook
            sentiment = analysis_results.get('sentiment_analysis', {})
            sentiment_score = (sentiment.get('sentiment_report', {})
                             .get('overall_market_sentiment', {})
                             .get('score', 0))
            
            if abs(sentiment_score) > 0.2:
                sentiment_direction = "positive" if sentiment_score > 0 else "negative"
                outlook_factors.append(f"Social sentiment is {sentiment_direction}")
            
            # Arbitrage outlook
            arbitrage = analysis_results.get('arbitrage_opportunities', {})
            if arbitrage.get('summary', {}).get('total_simple_opportunities', 0) > 5:
                outlook_factors.append("Multiple arbitrage opportunities suggest market inefficiencies")
            
            # DeFi outlook
            defi = analysis_results.get('defi_opportunities', {})
            avg_yield = np.mean([
                opp.yield_pct for opps in defi.get('yield_opportunities', {}).values()
                for opp in opps[:5]  # Top 5 per category
            ]) if defi.get('yield_opportunities') else 0
            
            if avg_yield > 10:
                outlook_factors.append(f"DeFi yields averaging {avg_yield:.1f}% indicate attractive earning opportunities")
            
            # Combine outlook factors
            if outlook_factors:
                return ". ".join(outlook_factors) + "."
            else:
                return "Market conditions appear neutral with mixed signals across different analysis modules."
                
        except Exception as e:
            return "Unable to generate market outlook due to analysis error."
    
    def _calculate_avg_apy_by_risk(self, yield_opportunities: Dict[str, List]) -> Dict[str, float]:
        """Calculate average APY by risk level."""
        risk_yields = {'LOW': [], 'MEDIUM': [], 'HIGH': [], 'EXTREME': []}
        
        for category, opportunities in yield_opportunities.items():
            for opp in opportunities:
                risk_yields[opp.risk_level].append(opp.yield_pct)
        
        return {
            risk: np.mean(yields) if yields else 0
            for risk, yields in risk_yields.items()
        }
    
    def _analyze_protocol_distribution(self, yield_opportunities: Dict[str, List]) -> Dict[str, int]:
        """Analyze distribution of opportunities by protocol."""
        protocol_counts = {}
        
        for category, opportunities in yield_opportunities.items():
            for opp in opportunities:
                protocol_counts[opp.protocol] = protocol_counts.get(opp.protocol, 0) + 1
        
        return protocol_counts
    
    async def get_real_time_market_intelligence(self) -> Dict[str, Any]:
        """Get real-time market intelligence summary."""
        intelligence = {
            'timestamp': datetime.now().isoformat(),
            'market_alerts': [],
            'trending_opportunities': [],
            'risk_warnings': [],
            'performance_summary': {}
        }
        
        # Check for immediate opportunities from cache
        if self.analysis_cache.get('arbitrage'):
            arb_opps = self.analysis_cache['arbitrage'].get('simple_arbitrage', [])
            for opp in arb_opps[:3]:
                if opp.get('spread_pct', 0) > 3:
                    intelligence['market_alerts'].append({
                        'type': 'arbitrage',
                        'message': f"High spread detected: {opp['symbol']} - {opp['spread_pct']:.1f}%",
                        'urgency': 'high'
                    })
        
        # Check sentiment alerts
        if self.analysis_cache.get('sentiment'):
            sentiment_data = self.analysis_cache['sentiment'].get('sentiment_data', {})
            for symbol, data in sentiment_data.items():
                if abs(data.get('sentiment_score', 0)) > 0.5 and data.get('volume', 0) > 500:
                    sentiment_type = "very positive" if data['sentiment_score'] > 0 else "very negative"
                    intelligence['market_alerts'].append({
                        'type': 'sentiment',
                        'message': f"{symbol} showing {sentiment_type} sentiment with high volume",
                        'urgency': 'medium'
                    })
        
        return intelligence

# Global master analyzer instance
master_analyzer = MasterCryptoAnalyzer()