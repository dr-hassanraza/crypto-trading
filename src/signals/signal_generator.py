import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from enum import Enum

from src.analyzers.technical_indicators import TechnicalAnalyzer
from src.analyzers.ai_analysis import AIMarketAnalyzer
from config.config import Config

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class SignalConfidence(Enum):
    VERY_LOW = "VERY_LOW"    # 0-20%
    LOW = "LOW"              # 20-40%
    MEDIUM = "MEDIUM"        # 40-70%
    HIGH = "HIGH"           # 70-85%
    VERY_HIGH = "VERY_HIGH" # 85%+

class AdvancedSignalGenerator:
    def __init__(self):
        self.config = Config()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = AIMarketAnalyzer()
        self.signal_history = {}  # Track signal history for each crypto
        
    def generate_comprehensive_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signals from all available data."""
        crypto_id = market_data.get('crypto_id')
        
        # Get technical analysis signals
        technical_signals = self._generate_technical_signals(market_data)
        
        # Get AI-powered signals  
        ai_signals = self._generate_ai_signals(market_data)
        
        # Get sentiment-based signals
        sentiment_signals = self._generate_sentiment_signals(market_data)
        
        # Get momentum signals
        momentum_signals = self._generate_momentum_signals(market_data)
        
        # Get volume-based signals
        volume_signals = self._generate_volume_signals(market_data)
        
        # Combine all signals with weighted scoring
        combined_signal = self._combine_signals({
            'technical': technical_signals,
            'ai': ai_signals,
            'sentiment': sentiment_signals,
            'momentum': momentum_signals,
            'volume': volume_signals
        })
        
        # Apply risk management filters
        filtered_signal = self._apply_risk_filters(combined_signal, market_data)
        
        # Generate final recommendation with confidence
        final_signal = self._generate_final_recommendation(filtered_signal, market_data)
        
        # Update signal history
        self._update_signal_history(crypto_id, final_signal)
        
        return {
            'crypto_id': crypto_id,
            'symbol': market_data.get('symbol'),
            'final_signal': final_signal,
            'component_signals': {
                'technical': technical_signals,
                'ai': ai_signals,
                'sentiment': sentiment_signals,
                'momentum': momentum_signals,
                'volume': volume_signals
            },
            'signal_strength': combined_signal.get('strength', 0),
            'confidence_score': final_signal.get('confidence_score', 0),
            'timestamp': datetime.now(),
            'data_quality': market_data.get('data_quality', {}),
            'signal_id': f"{crypto_id}_{int(datetime.now().timestamp())}"
        }
    
    def _generate_technical_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on technical indicators."""
        signals = []
        signal_strength = 0
        confidence = 0
        
        ohlcv_data = market_data.get('ohlcv_data', {})
        
        for timeframe, df in ohlcv_data.items():
            if df.empty or len(df) < 50:
                continue
                
            # Calculate technical indicators
            df_with_indicators = self.technical_analyzer.calculate_all_indicators(df)
            
            # Get signal strength from technical analyzer
            tech_signal = self.technical_analyzer.get_signal_strength(df_with_indicators)
            
            # Weight by timeframe importance
            timeframe_weight = self._get_timeframe_weight(timeframe)
            weighted_strength = tech_signal['strength'] * timeframe_weight
            
            signals.append({
                'timeframe': timeframe,
                'signal': tech_signal['direction'],
                'strength': tech_signal['strength'],
                'confidence': tech_signal['confidence'],
                'weight': timeframe_weight
            })
            
            signal_strength += weighted_strength
            confidence += tech_signal['confidence'] * timeframe_weight
        
        # Detect patterns
        patterns = {}
        if ohlcv_data.get('1d') is not None and not ohlcv_data['1d'].empty:
            df_1d = self.technical_analyzer.calculate_all_indicators(ohlcv_data['1d'])
            patterns = self.technical_analyzer.detect_patterns(df_1d)
        
        # Adjust for patterns
        pattern_adjustment = self._calculate_pattern_adjustment(patterns)
        signal_strength += pattern_adjustment
        
        return {
            'primary_signal': self._strength_to_signal(signal_strength),
            'strength': round(signal_strength, 3),
            'confidence': round(confidence / len(signals) if signals else 0, 1),
            'signals_by_timeframe': signals,
            'detected_patterns': patterns,
            'pattern_adjustment': pattern_adjustment
        }
    
    def _generate_ai_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on AI analysis."""
        try:
            ai_analysis = self.ai_analyzer.analyze_comprehensive_market_data(market_data)
            
            ai_signals = ai_analysis.get('signals', {})
            primary_signal = ai_signals.get('primary_signal', 'HOLD')
            signal_strength = ai_signals.get('signal_strength', 0)
            confidence = ai_analysis.get('confidence_score', 0)
            
            return {
                'primary_signal': primary_signal,
                'strength': signal_strength,
                'confidence': confidence,
                'ai_outlook': ai_analysis.get('ai_analysis', {}).get('ai_insight', {}).get('overall_outlook'),
                'risk_assessment': ai_analysis.get('risk_assessment', {}),
                'price_scenarios': ai_analysis.get('price_scenarios', {}),
                'key_levels': ai_signals.get('key_levels', {})
            }
        except Exception as e:
            logging.error(f"Error generating AI signals: {e}")
            return {
                'primary_signal': 'HOLD',
                'strength': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    def _generate_sentiment_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on market sentiment."""
        sentiment_data = market_data.get('news_sentiment', {})
        fear_greed = market_data.get('coingecko', {}).get('fear_greed_index', {})
        
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        article_count = sentiment_data.get('article_count', 0)
        fg_value = int(fear_greed.get('value', 50)) if fear_greed.get('value') else 50
        
        # Calculate sentiment signal strength
        strength = 0
        confidence = 0
        
        # News sentiment component
        if sentiment_score > 0.2:
            strength += 0.3
        elif sentiment_score > 0.1:
            strength += 0.15
        elif sentiment_score < -0.2:
            strength -= 0.3
        elif sentiment_score < -0.1:
            strength -= 0.15
        
        # Fear & Greed component  
        if fg_value <= 25:  # Extreme Fear - potential buy
            strength += 0.25
        elif fg_value <= 45:  # Fear - mild buy
            strength += 0.1
        elif fg_value >= 75:  # Extreme Greed - potential sell
            strength -= 0.25
        elif fg_value >= 55:  # Greed - mild sell
            strength -= 0.1
        
        # Confidence based on data availability
        if article_count > 10:
            confidence = 70
        elif article_count > 5:
            confidence = 50
        else:
            confidence = 30
        
        return {
            'primary_signal': self._strength_to_signal(strength),
            'strength': round(strength, 3),
            'confidence': confidence,
            'news_sentiment_score': sentiment_score,
            'fear_greed_index': fg_value,
            'article_count': article_count,
            'sentiment_interpretation': self._interpret_sentiment(sentiment_score, fg_value)
        }
    
    def _generate_momentum_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on price momentum."""
        coingecko_data = market_data.get('coingecko', {}).get('market_data', [])
        
        if not coingecko_data:
            return {'primary_signal': 'HOLD', 'strength': 0, 'confidence': 0}
        
        coin_data = coingecko_data[0]
        
        # Price changes across different timeframes
        price_1h = coin_data.get('price_change_percentage_1h_in_currency', 0)
        price_24h = coin_data.get('price_change_percentage_24h', 0)
        price_7d = coin_data.get('price_change_percentage_7d_in_currency', 0)
        price_30d = coin_data.get('price_change_percentage_30d_in_currency', 0)
        
        # Calculate momentum score
        momentum_score = 0
        
        # Short-term momentum (1h, 24h)
        if price_1h > 2:
            momentum_score += 0.2
        elif price_1h > 1:
            momentum_score += 0.1
        elif price_1h < -2:
            momentum_score -= 0.2
        elif price_1h < -1:
            momentum_score -= 0.1
        
        if price_24h > 5:
            momentum_score += 0.3
        elif price_24h > 2:
            momentum_score += 0.15
        elif price_24h < -5:
            momentum_score -= 0.3
        elif price_24h < -2:
            momentum_score -= 0.15
        
        # Medium-term momentum (7d, 30d)
        if price_7d > 10:
            momentum_score += 0.25
        elif price_7d > 5:
            momentum_score += 0.1
        elif price_7d < -10:
            momentum_score -= 0.25
        elif price_7d < -5:
            momentum_score -= 0.1
        
        if price_30d > 20:
            momentum_score += 0.2
        elif price_30d > 10:
            momentum_score += 0.1
        elif price_30d < -20:
            momentum_score -= 0.2
        elif price_30d < -10:
            momentum_score -= 0.1
        
        # Momentum consistency check
        momentum_directions = [
            1 if price_1h > 0 else -1 if price_1h < 0 else 0,
            1 if price_24h > 0 else -1 if price_24h < 0 else 0,
            1 if price_7d > 0 else -1 if price_7d < 0 else 0
        ]
        
        consistency = abs(sum(momentum_directions))
        confidence = min(consistency * 25 + 25, 85)  # 25-85% based on consistency
        
        return {
            'primary_signal': self._strength_to_signal(momentum_score),
            'strength': round(momentum_score, 3),
            'confidence': confidence,
            'price_changes': {
                '1h': price_1h,
                '24h': price_24h,
                '7d': price_7d,
                '30d': price_30d
            },
            'momentum_consistency': consistency,
            'momentum_interpretation': self._interpret_momentum(momentum_score, consistency)
        }
    
    def _generate_volume_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on volume analysis."""
        ohlcv_1d = market_data.get('ohlcv_data', {}).get('1d', pd.DataFrame())
        coingecko_data = market_data.get('coingecko', {}).get('market_data', [])
        
        if ohlcv_1d.empty or not coingecko_data:
            return {'primary_signal': 'HOLD', 'strength': 0, 'confidence': 0}
        
        # Current volume vs average
        current_volume = ohlcv_1d['volume'].iloc[-1]
        avg_volume_20 = ohlcv_1d['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # Volume trend
        volume_sma_5 = ohlcv_1d['volume'].rolling(5).mean()
        volume_trend = volume_sma_5.iloc[-1] / volume_sma_5.iloc[-6] if len(volume_sma_5) > 5 else 1
        
        # Price-Volume relationship
        price_change = ohlcv_1d['close'].pct_change().iloc[-1]
        volume_change = ohlcv_1d['volume'].pct_change().iloc[-1]
        
        strength = 0
        
        # High volume with price movement (confirmation)
        if volume_ratio > 2 and abs(price_change) > 0.02:  # >2x volume and >2% price change
            if price_change > 0:
                strength += 0.4  # Bullish confirmation
            else:
                strength -= 0.4  # Bearish confirmation
        elif volume_ratio > 1.5 and abs(price_change) > 0.01:
            if price_change > 0:
                strength += 0.2
            else:
                strength -= 0.2
        
        # Volume trend
        if volume_trend > 1.2:
            strength += 0.1
        elif volume_trend < 0.8:
            strength -= 0.1
        
        # Low volume warning (reduces confidence)
        confidence_penalty = 0
        if volume_ratio < 0.5:
            confidence_penalty = 20
            
        base_confidence = min(volume_ratio * 30, 70)
        final_confidence = max(base_confidence - confidence_penalty, 20)
        
        return {
            'primary_signal': self._strength_to_signal(strength),
            'strength': round(strength, 3),
            'confidence': final_confidence,
            'volume_ratio': round(volume_ratio, 2),
            'volume_trend': round(volume_trend, 2),
            'volume_interpretation': self._interpret_volume(volume_ratio, volume_trend, price_change)
        }
    
    def _combine_signals(self, signal_components: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine all signal components with weighted averaging."""
        weights = {
            'technical': 0.35,  # Highest weight for technical analysis
            'ai': 0.25,         # High weight for AI insights
            'momentum': 0.2,    # Medium weight for momentum
            'sentiment': 0.1,   # Lower weight for sentiment  
            'volume': 0.1       # Lower weight for volume
        }
        
        total_strength = 0
        total_confidence = 0
        signal_count = 0
        
        component_summary = {}
        
        for component, signals in signal_components.items():
            if signals and signals.get('strength') is not None:
                weight = weights.get(component, 0.1)
                strength = signals.get('strength', 0)
                confidence = signals.get('confidence', 0)
                
                # Apply confidence weighting
                confidence_factor = confidence / 100
                weighted_strength = strength * weight * confidence_factor
                weighted_confidence = confidence * weight
                
                total_strength += weighted_strength
                total_confidence += weighted_confidence
                signal_count += 1
                
                component_summary[component] = {
                    'signal': signals.get('primary_signal', 'HOLD'),
                    'strength': strength,
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_contribution': weighted_strength
                }
        
        # Normalize confidence
        avg_confidence = total_confidence / sum(weights.values()) if signal_count > 0 else 0
        
        return {
            'combined_strength': round(total_strength, 3),
            'combined_confidence': round(avg_confidence, 1),
            'primary_signal': self._strength_to_signal(total_strength),
            'component_summary': component_summary,
            'signal_components_used': signal_count
        }
    
    def _apply_risk_filters(self, combined_signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management filters to signals."""
        filtered_signal = combined_signal.copy()
        risk_flags = []
        
        # Data quality filter
        data_quality = market_data.get('data_quality', {}).get('score', 0)
        if data_quality < 50:
            risk_flags.append("LOW_DATA_QUALITY")
            filtered_signal['combined_confidence'] *= 0.7  # Reduce confidence
        
        # Volatility filter
        coingecko_data = market_data.get('coingecko', {}).get('market_data', [])
        if coingecko_data:
            price_change_24h = abs(coingecko_data[0].get('price_change_percentage_24h', 0))
            if price_change_24h > 20:  # Very high volatility
                risk_flags.append("HIGH_VOLATILITY")
                # Reduce signal strength for buy signals in high volatility
                if filtered_signal['combined_strength'] > 0:
                    filtered_signal['combined_strength'] *= 0.8
        
        # Market cap filter
        if coingecko_data:
            market_cap = coingecko_data[0].get('market_cap', 0)
            if market_cap < 1e8:  # Less than $100M market cap
                risk_flags.append("SMALL_MARKET_CAP")
                filtered_signal['combined_confidence'] *= 0.8
        
        # Volume filter
        ohlcv_1d = market_data.get('ohlcv_data', {}).get('1d', pd.DataFrame())
        if not ohlcv_1d.empty:
            current_volume = ohlcv_1d['volume'].iloc[-1]
            avg_volume = ohlcv_1d['volume'].rolling(20).mean().iloc[-1]
            if current_volume < avg_volume * 0.3:  # Very low volume
                risk_flags.append("LOW_VOLUME")
                filtered_signal['combined_confidence'] *= 0.6
        
        # Signal history filter (prevent whipsaws)
        crypto_id = market_data.get('crypto_id')
        if crypto_id in self.signal_history:
            recent_signals = self.signal_history[crypto_id][-3:]  # Last 3 signals
            if len(recent_signals) >= 2:
                signal_changes = sum(1 for i in range(1, len(recent_signals)) 
                                   if recent_signals[i]['signal'] != recent_signals[i-1]['signal'])
                if signal_changes >= 2:  # Too many signal changes
                    risk_flags.append("SIGNAL_INSTABILITY")
                    filtered_signal['combined_confidence'] *= 0.7
        
        filtered_signal['risk_flags'] = risk_flags
        return filtered_signal
    
    def _generate_final_recommendation(self, filtered_signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendation with all details."""
        strength = filtered_signal['combined_strength']
        confidence = filtered_signal['combined_confidence']
        
        # Determine final signal with thresholds
        if strength >= 0.6 and confidence >= 70:
            final_signal = SignalType.STRONG_BUY.value
        elif strength >= 0.3 and confidence >= 50:
            final_signal = SignalType.BUY.value
        elif strength <= -0.6 and confidence >= 70:
            final_signal = SignalType.STRONG_SELL.value
        elif strength <= -0.3 and confidence >= 50:
            final_signal = SignalType.SELL.value
        else:
            final_signal = SignalType.HOLD.value
        
        # Confidence classification
        confidence_level = self._classify_confidence(confidence)
        
        # Calculate position sizing recommendation
        position_size = self._calculate_position_size(strength, confidence, filtered_signal.get('risk_flags', []))
        
        # Set stop-loss and take-profit levels
        current_price = self._get_current_price(market_data)
        stop_loss, take_profit = self._calculate_stop_take_levels(final_signal, current_price, strength)
        
        return {
            'signal': final_signal,
            'strength': round(strength, 3),
            'confidence_score': round(confidence, 1),
            'confidence_level': confidence_level.value,
            'position_size': position_size,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': self._calculate_risk_reward(current_price, stop_loss, take_profit),
            'risk_flags': filtered_signal.get('risk_flags', []),
            'signal_reasons': self._generate_signal_reasons(filtered_signal),
            'time_horizon': self._estimate_time_horizon(strength, confidence),
            'market_context': self._get_market_context(market_data)
        }
    
    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get weight for different timeframes."""
        weights = {
            '1h': 0.1,
            '4h': 0.3,
            '1d': 0.6
        }
        return weights.get(timeframe, 0.1)
    
    def _calculate_pattern_adjustment(self, patterns: Dict[str, bool]) -> float:
        """Calculate signal adjustment based on detected patterns."""
        adjustment = 0
        
        if patterns.get('golden_cross'):
            adjustment += 0.2
        elif patterns.get('death_cross'):
            adjustment -= 0.2
            
        if patterns.get('bullish_divergence'):
            adjustment += 0.15
        elif patterns.get('bearish_divergence'):
            adjustment -= 0.15
            
        if patterns.get('squeeze'):
            adjustment += 0.1  # Potential breakout
        
        return adjustment
    
    def _strength_to_signal(self, strength: float) -> str:
        """Convert signal strength to signal type."""
        if strength >= 0.6:
            return SignalType.STRONG_BUY.value
        elif strength >= 0.2:
            return SignalType.BUY.value
        elif strength <= -0.6:
            return SignalType.STRONG_SELL.value
        elif strength <= -0.2:
            return SignalType.SELL.value
        else:
            return SignalType.HOLD.value
    
    def _interpret_sentiment(self, sentiment_score: float, fg_value: int) -> str:
        """Interpret sentiment data."""
        if sentiment_score > 0.2 and fg_value < 30:
            return "Positive news with extreme fear - potential opportunity"
        elif sentiment_score < -0.2 and fg_value > 70:
            return "Negative news with extreme greed - high risk"
        elif sentiment_score > 0.1:
            return "Generally positive sentiment"
        elif sentiment_score < -0.1:
            return "Generally negative sentiment"
        else:
            return "Neutral sentiment"
    
    def _interpret_momentum(self, momentum_score: float, consistency: int) -> str:
        """Interpret momentum analysis."""
        if momentum_score > 0.5:
            return f"Strong upward momentum (consistency: {consistency}/3)"
        elif momentum_score > 0.2:
            return f"Moderate upward momentum (consistency: {consistency}/3)"
        elif momentum_score < -0.5:
            return f"Strong downward momentum (consistency: {consistency}/3)"
        elif momentum_score < -0.2:
            return f"Moderate downward momentum (consistency: {consistency}/3)"
        else:
            return f"Neutral momentum (consistency: {consistency}/3)"
    
    def _interpret_volume(self, volume_ratio: float, volume_trend: float, price_change: float) -> str:
        """Interpret volume analysis."""
        if volume_ratio > 2:
            if price_change > 0:
                return "High volume confirms upward movement"
            else:
                return "High volume confirms downward movement"
        elif volume_ratio > 1.5:
            return "Above-average volume supports price action"
        elif volume_ratio < 0.5:
            return "Low volume - price movement lacks conviction"
        else:
            return "Normal volume levels"
    
    def _classify_confidence(self, confidence: float) -> SignalConfidence:
        """Classify confidence score into categories."""
        if confidence >= 85:
            return SignalConfidence.VERY_HIGH
        elif confidence >= 70:
            return SignalConfidence.HIGH
        elif confidence >= 40:
            return SignalConfidence.MEDIUM
        elif confidence >= 20:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW
    
    def _calculate_position_size(self, strength: float, confidence: float, risk_flags: List[str]) -> str:
        """Calculate recommended position size."""
        base_size = abs(strength) * confidence / 100
        
        # Adjust for risk flags
        risk_penalty = len(risk_flags) * 0.1
        adjusted_size = max(base_size - risk_penalty, 0.01)
        
        if adjusted_size >= 0.08:
            return "LARGE (5-8%)"
        elif adjusted_size >= 0.05:
            return "MEDIUM (3-5%)"
        elif adjusted_size >= 0.02:
            return "SMALL (1-3%)"
        else:
            return "MICRO (<1%)"
    
    def _get_current_price(self, market_data: Dict[str, Any]) -> Optional[float]:
        """Extract current price from market data."""
        coingecko_data = market_data.get('coingecko', {}).get('market_data', [])
        if coingecko_data:
            return coingecko_data[0].get('current_price')
        return None
    
    def _calculate_stop_take_levels(self, signal: str, current_price: Optional[float], strength: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop-loss and take-profit levels."""
        if not current_price:
            return None, None
        
        # Dynamic stop-loss based on signal strength and type
        if signal in [SignalType.BUY.value, SignalType.STRONG_BUY.value]:
            stop_loss_pct = max(0.05, 0.15 - abs(strength) * 0.1)  # 5-10% stop loss
            take_profit_pct = min(0.3, abs(strength) * 0.5)        # Up to 30% take profit
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            
        elif signal in [SignalType.SELL.value, SignalType.STRONG_SELL.value]:
            stop_loss_pct = max(0.05, 0.15 - abs(strength) * 0.1)  # 5-10% stop loss
            take_profit_pct = min(0.3, abs(strength) * 0.5)        # Up to 30% take profit
            
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            return None, None
        
        return round(stop_loss, 6), round(take_profit, 6)
    
    def _calculate_risk_reward(self, entry: Optional[float], stop_loss: Optional[float], take_profit: Optional[float]) -> Optional[str]:
        """Calculate risk-reward ratio."""
        if not all([entry, stop_loss, take_profit]):
            return None
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk > 0:
            ratio = reward / risk
            return f"1:{ratio:.2f}"
        
        return None
    
    def _generate_signal_reasons(self, filtered_signal: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasons for the signal."""
        reasons = []
        component_summary = filtered_signal.get('component_summary', {})
        
        for component, data in component_summary.items():
            contribution = data.get('weighted_contribution', 0)
            if abs(contribution) > 0.1:  # Significant contribution
                direction = "positive" if contribution > 0 else "negative"
                reasons.append(f"{component.title()} analysis shows {direction} signals")
        
        risk_flags = filtered_signal.get('risk_flags', [])
        if risk_flags:
            reasons.append(f"Risk factors identified: {', '.join(risk_flags)}")
        
        return reasons
    
    def _estimate_time_horizon(self, strength: float, confidence: float) -> str:
        """Estimate optimal time horizon for the signal."""
        if abs(strength) > 0.5 and confidence > 70:
            return "1-2 weeks"
        elif abs(strength) > 0.3 and confidence > 50:
            return "3-7 days" 
        else:
            return "1-3 days"
    
    def _get_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant market context information."""
        fear_greed = market_data.get('coingecko', {}).get('fear_greed_index', {})
        coingecko_data = market_data.get('coingecko', {}).get('market_data', [])
        
        context = {}
        if fear_greed:
            context['fear_greed_index'] = fear_greed.get('value')
            context['market_sentiment'] = fear_greed.get('value_classification')
        
        if coingecko_data:
            context['market_cap_rank'] = coingecko_data[0].get('market_cap_rank')
            context['24h_volume'] = coingecko_data[0].get('total_volume')
        
        return context
    
    def _update_signal_history(self, crypto_id: str, signal_data: Dict[str, Any]) -> None:
        """Update signal history for tracking."""
        if crypto_id not in self.signal_history:
            self.signal_history[crypto_id] = []
        
        # Keep only last 10 signals
        history_entry = {
            'timestamp': datetime.now(),
            'signal': signal_data.get('signal'),
            'strength': signal_data.get('strength'),
            'confidence': signal_data.get('confidence_score')
        }
        
        self.signal_history[crypto_id].append(history_entry)
        self.signal_history[crypto_id] = self.signal_history[crypto_id][-10:]