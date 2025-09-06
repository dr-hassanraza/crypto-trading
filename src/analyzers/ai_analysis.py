import openai
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from config.config import Config

class AIMarketAnalyzer:
    def __init__(self):
        self.config = Config()
        if self.config.OPENAI_API_KEY:
            openai.api_key = self.config.OPENAI_API_KEY
        else:
            print("Warning: OpenAI API key not found")
    
    def analyze_comprehensive_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive AI analysis of market data."""
        
        # Prepare data summary for AI analysis
        data_summary = self._prepare_data_summary(market_data)
        
        # Generate AI analysis
        ai_analysis = self._get_ai_market_insight(data_summary)
        
        # Generate specific signals
        signals = self._generate_ai_signals(market_data, ai_analysis)
        
        # Risk assessment
        risk_assessment = self._assess_market_risk(market_data, ai_analysis)
        
        # Price targets and scenarios
        scenarios = self._generate_price_scenarios(market_data, ai_analysis)
        
        return {
            'crypto_id': market_data.get('crypto_id'),
            'symbol': market_data.get('symbol'),
            'ai_analysis': ai_analysis,
            'signals': signals,
            'risk_assessment': risk_assessment,
            'price_scenarios': scenarios,
            'confidence_score': self._calculate_confidence_score(market_data, ai_analysis),
            'timestamp': datetime.now(),
            'data_quality': market_data.get('data_quality', {})
        }
    
    def _prepare_data_summary(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a concise summary of all market data for AI analysis."""
        summary = {
            'crypto_info': {
                'id': market_data.get('crypto_id'),
                'symbol': market_data.get('symbol')
            }
        }
        
        # CoinGecko market data
        coingecko = market_data.get('coingecko', {}).get('market_data', [])
        if coingecko and len(coingecko) > 0:
            coin_data = coingecko[0]
            summary['current_metrics'] = {
                'price': coin_data.get('current_price'),
                'market_cap': coin_data.get('market_cap'),
                'volume_24h': coin_data.get('total_volume'),
                'price_change_1h': coin_data.get('price_change_percentage_1h_in_currency'),
                'price_change_24h': coin_data.get('price_change_percentage_24h'),
                'price_change_7d': coin_data.get('price_change_percentage_7d_in_currency'),
                'price_change_30d': coin_data.get('price_change_percentage_30d_in_currency'),
                'market_cap_rank': coin_data.get('market_cap_rank'),
                'ath': coin_data.get('ath'),
                'atl': coin_data.get('atl'),
                'ath_change_percentage': coin_data.get('ath_change_percentage'),
                'atl_change_percentage': coin_data.get('atl_change_percentage')
            }
        
        # Technical indicators (from OHLCV data)
        ohlcv_1d = market_data.get('ohlcv_data', {}).get('1d', pd.DataFrame())
        if not ohlcv_1d.empty and len(ohlcv_1d) > 20:
            # Calculate simple indicators for summary
            latest_close = ohlcv_1d['close'].iloc[-1]
            sma_20 = ohlcv_1d['close'].rolling(20).mean().iloc[-1]
            sma_50 = ohlcv_1d['close'].rolling(50).mean().iloc[-1] if len(ohlcv_1d) > 50 else None
            
            # RSI calculation
            delta = ohlcv_1d['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volume trend
            volume_sma = ohlcv_1d['volume'].rolling(20).mean()
            current_volume = ohlcv_1d['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
            
            summary['technical_indicators'] = {
                'price_vs_sma_20': (latest_close - sma_20) / sma_20 * 100 if sma_20 else None,
                'price_vs_sma_50': (latest_close - sma_50) / sma_50 * 100 if sma_50 else None,
                'rsi': rsi.iloc[-1] if not rsi.empty else None,
                'volume_ratio': volume_ratio,
                'recent_high': ohlcv_1d['high'].rolling(30).max().iloc[-1],
                'recent_low': ohlcv_1d['low'].rolling(30).min().iloc[-1],
                'volatility': ohlcv_1d['close'].pct_change().rolling(20).std().iloc[-1] * 100
            }
        
        # News sentiment
        news_sentiment = market_data.get('news_sentiment', {})
        if news_sentiment:
            summary['news_sentiment'] = {
                'sentiment_score': news_sentiment.get('sentiment_score'),
                'article_count': news_sentiment.get('article_count'),
                'positive_ratio': news_sentiment.get('positive_ratio'),
                'negative_ratio': news_sentiment.get('negative_ratio'),
                'recent_headlines': news_sentiment.get('recent_headlines', [])[:3]
            }
        
        # Market context
        fear_greed = market_data.get('coingecko', {}).get('fear_greed_index')
        if fear_greed:
            summary['market_context'] = {
                'fear_greed_index': fear_greed.get('value'),
                'fear_greed_classification': fear_greed.get('value_classification')
            }
        
        return summary
    
    def _get_ai_market_insight(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered market insight using OpenAI."""
        if not self.config.OPENAI_API_KEY:
            return self._generate_fallback_analysis(data_summary)
        
        # Create prompt for AI analysis
        prompt = self._create_analysis_prompt(data_summary)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an expert cryptocurrency market analyst with deep knowledge of technical analysis, fundamental analysis, and market psychology. 
                    Analyze the provided market data and give comprehensive insights including trend analysis, key support/resistance levels, 
                    potential catalysts, and market outlook. Be specific with price levels and timeframes where possible.
                    
                    Your analysis should include:
                    1. Short-term outlook (1-7 days)
                    2. Medium-term outlook (1-4 weeks) 
                    3. Key technical levels to watch
                    4. Risk factors and potential catalysts
                    5. Overall market sentiment assessment
                    
                    Provide your response in a structured JSON format."""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse AI response
            ai_content = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to text analysis
            try:
                ai_analysis = json.loads(ai_content)
            except json.JSONDecodeError:
                ai_analysis = {
                    'raw_analysis': ai_content,
                    'structured_analysis': self._parse_unstructured_response(ai_content)
                }
            
            return {
                'ai_insight': ai_analysis,
                'model_used': 'gpt-4',
                'tokens_used': response.usage.total_tokens,
                'generation_time': datetime.now()
            }
        
        except Exception as e:
            print(f"Error getting AI analysis: {e}")
            return self._generate_fallback_analysis(data_summary)
    
    def _create_analysis_prompt(self, data_summary: Dict[str, Any]) -> str:
        """Create detailed prompt for AI analysis."""
        crypto_info = data_summary.get('crypto_info', {})
        current_metrics = data_summary.get('current_metrics', {})
        technical = data_summary.get('technical_indicators', {})
        sentiment = data_summary.get('news_sentiment', {})
        market_context = data_summary.get('market_context', {})
        
        prompt = f"""
        Analyze the following cryptocurrency market data:
        
        CRYPTOCURRENCY: {crypto_info.get('id', 'Unknown').upper()} ({crypto_info.get('symbol', 'N/A')})
        
        CURRENT MARKET METRICS:
        - Current Price: ${current_metrics.get('price', 'N/A'):,.2f}
        - Market Cap: ${current_metrics.get('market_cap', 0):,.0f}
        - 24h Volume: ${current_metrics.get('volume_24h', 0):,.0f}
        - Price Changes: 1h: {current_metrics.get('price_change_1h', 0):.2f}%, 24h: {current_metrics.get('price_change_24h', 0):.2f}%, 7d: {current_metrics.get('price_change_7d', 0):.2f}%
        - Market Cap Rank: #{current_metrics.get('market_cap_rank', 'N/A')}
        - ATH: ${current_metrics.get('ath', 0):,.2f} (Change: {current_metrics.get('ath_change_percentage', 0):.2f}%)
        - ATL: ${current_metrics.get('atl', 0):,.2f} (Change: {current_metrics.get('atl_change_percentage', 0):.2f}%)
        
        TECHNICAL INDICATORS:
        - Price vs 20-day SMA: {technical.get('price_vs_sma_20', 0):.2f}%
        - Price vs 50-day SMA: {technical.get('price_vs_sma_50', 0):.2f}%
        - RSI: {technical.get('rsi', 0):.1f}
        - Volume Ratio (vs 20-day avg): {technical.get('volume_ratio', 1):.2f}x
        - Recent 30-day High: ${technical.get('recent_high', 0):,.2f}
        - Recent 30-day Low: ${technical.get('recent_low', 0):,.2f}
        - 20-day Volatility: {technical.get('volatility', 0):.2f}%
        
        NEWS SENTIMENT:
        - Sentiment Score: {sentiment.get('sentiment_score', 0):.3f} (-1 to 1 scale)
        - Article Count (7 days): {sentiment.get('article_count', 0)}
        - Positive News Ratio: {sentiment.get('positive_ratio', 0):.1%}
        - Negative News Ratio: {sentiment.get('negative_ratio', 0):.1%}
        - Recent Headlines: {', '.join(sentiment.get('recent_headlines', []))}
        
        MARKET CONTEXT:
        - Fear & Greed Index: {market_context.get('fear_greed_index', 'N/A')} ({market_context.get('fear_greed_classification', 'N/A')})
        
        Please provide a comprehensive analysis in JSON format with the following structure:
        {{
            "overall_outlook": "BULLISH/BEARISH/NEUTRAL",
            "confidence_level": "HIGH/MEDIUM/LOW",
            "short_term_analysis": {{
                "outlook": "1-7 day outlook",
                "key_levels": ["support and resistance levels"],
                "probability": "percentage"
            }},
            "medium_term_analysis": {{
                "outlook": "1-4 week outlook", 
                "price_targets": ["potential price levels"],
                "timeframe": "expected timeframe"
            }},
            "technical_summary": "technical analysis summary",
            "fundamental_factors": ["list of key factors"],
            "risk_factors": ["potential risks"],
            "catalysts": ["potential positive catalysts"],
            "recommended_action": "BUY/SELL/HOLD",
            "position_sizing": "suggested position size",
            "stop_loss": "suggested stop loss level",
            "take_profit": "suggested take profit level"
        }}
        """
        
        return prompt
    
    def _generate_fallback_analysis(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic analysis when AI is not available."""
        current_metrics = data_summary.get('current_metrics', {})
        technical = data_summary.get('technical_indicators', {})
        sentiment = data_summary.get('news_sentiment', {})
        
        # Basic rule-based analysis
        outlook = "NEUTRAL"
        confidence = "LOW"
        
        price_change_24h = current_metrics.get('price_change_24h', 0)
        rsi = technical.get('rsi', 50)
        sentiment_score = sentiment.get('sentiment_score', 0)
        
        # Simple scoring system
        score = 0
        if price_change_24h > 5:
            score += 2
        elif price_change_24h > 0:
            score += 1
        elif price_change_24h < -5:
            score -= 2
        elif price_change_24h < 0:
            score -= 1
        
        if rsi < 30:
            score += 1  # Oversold, potential buy
        elif rsi > 70:
            score -= 1  # Overbought, potential sell
        
        if sentiment_score > 0.1:
            score += 1
        elif sentiment_score < -0.1:
            score -= 1
        
        if score >= 2:
            outlook = "BULLISH"
            confidence = "MEDIUM" if score >= 3 else "LOW"
        elif score <= -2:
            outlook = "BEARISH"
            confidence = "MEDIUM" if score <= -3 else "LOW"
        
        return {
            'ai_insight': {
                'overall_outlook': outlook,
                'confidence_level': confidence,
                'short_term_analysis': {
                    'outlook': f"Based on {price_change_24h:.2f}% 24h change and RSI of {rsi:.1f}",
                    'key_levels': [current_metrics.get('price', 0) * 0.95, current_metrics.get('price', 0) * 1.05],
                    'probability': "50%"
                },
                'technical_summary': f"RSI: {rsi:.1f}, Price trend: {'Up' if price_change_24h > 0 else 'Down'}",
                'recommended_action': "HOLD" if outlook == "NEUTRAL" else ("BUY" if outlook == "BULLISH" else "SELL"),
                'fallback_analysis': True
            },
            'model_used': 'rule_based_fallback',
            'generation_time': datetime.now()
        }
    
    def _generate_ai_signals(self, market_data: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on AI analysis and technical data."""
        ai_insight = ai_analysis.get('ai_insight', {})
        
        # Extract key information
        outlook = ai_insight.get('overall_outlook', 'NEUTRAL')
        confidence = ai_insight.get('confidence_level', 'LOW')
        recommended_action = ai_insight.get('recommended_action', 'HOLD')
        
        # Calculate signal strength
        strength_mapping = {
            ('BULLISH', 'HIGH'): 0.8,
            ('BULLISH', 'MEDIUM'): 0.6,
            ('BULLISH', 'LOW'): 0.4,
            ('BEARISH', 'HIGH'): -0.8,
            ('BEARISH', 'MEDIUM'): -0.6,
            ('BEARISH', 'LOW'): -0.4,
            ('NEUTRAL', 'HIGH'): 0.1,
            ('NEUTRAL', 'MEDIUM'): 0.0,
            ('NEUTRAL', 'LOW'): 0.0
        }
        
        signal_strength = strength_mapping.get((outlook, confidence), 0.0)
        
        # Determine entry/exit signals
        entry_signal = None
        exit_signal = None
        
        if recommended_action == 'BUY' and signal_strength > 0.5:
            entry_signal = 'STRONG_BUY'
        elif recommended_action == 'BUY' and signal_strength > 0.2:
            entry_signal = 'BUY'
        elif recommended_action == 'SELL' and signal_strength < -0.5:
            exit_signal = 'STRONG_SELL'
        elif recommended_action == 'SELL' and signal_strength < -0.2:
            exit_signal = 'SELL'
        
        return {
            'primary_signal': recommended_action,
            'signal_strength': signal_strength,
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
            'confidence_percentage': self._confidence_to_percentage(confidence),
            'stop_loss': ai_insight.get('stop_loss'),
            'take_profit': ai_insight.get('take_profit'),
            'position_sizing': ai_insight.get('position_sizing', 'SMALL'),
            'time_horizon': self._extract_time_horizon(ai_insight),
            'key_levels': {
                'support': ai_insight.get('short_term_analysis', {}).get('key_levels', [None])[0],
                'resistance': ai_insight.get('short_term_analysis', {}).get('key_levels', [None, None])[1] if len(ai_insight.get('short_term_analysis', {}).get('key_levels', [])) > 1 else None
            }
        }
    
    def _assess_market_risk(self, market_data: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various risk factors."""
        current_metrics = market_data.get('coingecko', {}).get('market_data', [{}])[0] if market_data.get('coingecko', {}).get('market_data') else {}
        technical = market_data.get('ohlcv_data', {}).get('1d', pd.DataFrame())
        ai_insight = ai_analysis.get('ai_insight', {})
        
        risk_factors = []
        risk_score = 0  # 0-100 scale
        
        # Volatility risk
        if not technical.empty and len(technical) > 20:
            volatility = technical['close'].pct_change().rolling(20).std().iloc[-1] * 100
            if volatility > 10:
                risk_factors.append(f"High volatility: {volatility:.1f}%")
                risk_score += 20
            elif volatility > 5:
                risk_factors.append(f"Moderate volatility: {volatility:.1f}%")
                risk_score += 10
        
        # Price distance from ATH
        ath_change = current_metrics.get('ath_change_percentage', 0)
        if ath_change < -50:
            risk_factors.append(f"Price {abs(ath_change):.1f}% below ATH")
            risk_score += 15
        elif ath_change < -20:
            risk_factors.append(f"Price {abs(ath_change):.1f}% below ATH")
            risk_score += 5
        
        # Volume analysis
        volume_ratio = technical.get('volume_ratio', 1) if hasattr(technical, 'get') else 1
        if volume_ratio < 0.5:
            risk_factors.append("Low trading volume")
            risk_score += 15
        
        # Market cap risk
        market_cap = current_metrics.get('market_cap', 0)
        if market_cap < 1e9:  # Less than $1B
            risk_factors.append("Small market cap")
            risk_score += 20
        elif market_cap < 1e10:  # Less than $10B
            risk_factors.append("Medium market cap")
            risk_score += 10
        
        # AI-identified risks
        ai_risks = ai_insight.get('risk_factors', [])
        if ai_risks:
            risk_factors.extend(ai_risks)
            risk_score += len(ai_risks) * 5
        
        # Risk level
        if risk_score <= 20:
            risk_level = "LOW"
        elif risk_score <= 40:
            risk_level = "MEDIUM"
        elif risk_score <= 60:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"
        
        return {
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_position_size': self._calculate_position_size(risk_score),
            'max_drawdown_estimate': f"{risk_score / 2:.1f}%",
            'time_horizon_risk': "SHORT" if risk_score > 50 else "MEDIUM"
        }
    
    def _generate_price_scenarios(self, market_data: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate different price scenarios."""
        current_price = market_data.get('coingecko', {}).get('market_data', [{}])[0].get('current_price', 0) if market_data.get('coingecko', {}).get('market_data') else 0
        ai_insight = ai_analysis.get('ai_insight', {})
        
        if current_price == 0:
            return {}
        
        # Extract AI price targets if available
        price_targets = ai_insight.get('medium_term_analysis', {}).get('price_targets', [])
        
        # Generate scenarios
        scenarios = {}
        
        # Bull case (30% probability)
        bull_multiplier = 1.5 if not price_targets else max(1.2, max(price_targets) / current_price if price_targets else 1.2)
        scenarios['bull_case'] = {
            'price_target': current_price * bull_multiplier,
            'probability': '30%',
            'timeframe': '1-3 months',
            'conditions': ['Strong market sentiment', 'Positive news flow', 'Technical breakout']
        }
        
        # Base case (40% probability)
        base_multiplier = 1.1 if not price_targets else (max(price_targets) + min(price_targets)) / (2 * current_price) if len(price_targets) >= 2 else 1.1
        scenarios['base_case'] = {
            'price_target': current_price * base_multiplier,
            'probability': '40%',
            'timeframe': '1-2 months', 
            'conditions': ['Market consolidation', 'Mixed signals', 'Range-bound trading']
        }
        
        # Bear case (30% probability)
        bear_multiplier = 0.8 if not price_targets else min(0.85, min(price_targets) / current_price if price_targets else 0.8)
        scenarios['bear_case'] = {
            'price_target': current_price * bear_multiplier,
            'probability': '30%',
            'timeframe': '2-4 weeks',
            'conditions': ['Market downturn', 'Negative catalysts', 'Technical breakdown']
        }
        
        return scenarios
    
    def _calculate_confidence_score(self, market_data: Dict[str, Any], ai_analysis: Dict[str, Any]) -> int:
        """Calculate overall confidence score (0-100)."""
        data_quality = market_data.get('data_quality', {}).get('score', 0)
        ai_confidence = ai_analysis.get('ai_insight', {}).get('confidence_level', 'LOW')
        
        confidence_mapping = {'LOW': 30, 'MEDIUM': 60, 'HIGH': 90}
        ai_score = confidence_mapping.get(ai_confidence, 30)
        
        # Weighted average
        final_score = (data_quality * 0.3) + (ai_score * 0.7)
        
        return int(final_score)
    
    def _confidence_to_percentage(self, confidence: str) -> int:
        """Convert confidence level to percentage."""
        mapping = {'LOW': 40, 'MEDIUM': 70, 'HIGH': 85}
        return mapping.get(confidence, 50)
    
    def _extract_time_horizon(self, ai_insight: Dict[str, Any]) -> str:
        """Extract time horizon from AI analysis."""
        short_term = ai_insight.get('short_term_analysis', {})
        medium_term = ai_insight.get('medium_term_analysis', {})
        
        if medium_term.get('timeframe'):
            return medium_term['timeframe']
        elif short_term:
            return "1-7 days"
        else:
            return "1-2 weeks"
    
    def _calculate_position_size(self, risk_score: int) -> str:
        """Calculate recommended position size based on risk."""
        if risk_score <= 20:
            return "LARGE (5-10%)"
        elif risk_score <= 40:
            return "MEDIUM (2-5%)"
        elif risk_score <= 60:
            return "SMALL (1-2%)"
        else:
            return "MICRO (<1%)"
    
    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """Parse unstructured AI response into structured format."""
        # Basic parsing for unstructured responses
        lines = response.split('\n')
        
        parsed = {
            'overall_outlook': 'NEUTRAL',
            'confidence_level': 'LOW',
            'technical_summary': response[:200] + '...' if len(response) > 200 else response,
            'recommended_action': 'HOLD'
        }
        
        # Simple keyword detection
        response_lower = response.lower()
        if any(word in response_lower for word in ['bullish', 'buy', 'positive', 'upward']):
            parsed['overall_outlook'] = 'BULLISH'
            parsed['recommended_action'] = 'BUY'
        elif any(word in response_lower for word in ['bearish', 'sell', 'negative', 'downward']):
            parsed['overall_outlook'] = 'BEARISH' 
            parsed['recommended_action'] = 'SELL'
        
        if any(word in response_lower for word in ['high confidence', 'strong', 'certain']):
            parsed['confidence_level'] = 'HIGH'
        elif any(word in response_lower for word in ['medium', 'moderate']):
            parsed['confidence_level'] = 'MEDIUM'
        
        return parsed