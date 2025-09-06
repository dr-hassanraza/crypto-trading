"""
Advanced Market Classification and Opportunity Identification System

Real-time market regime classification and opportunity detection using
clustering, Bayesian inference, and machine learning techniques.
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async
from .clustering_engine import clustering_engine, ClusterResult
from .feature_pipeline import feature_pipeline, FeaturePipelineConfig
from .bayesian_framework import bayesian_framework, BayesianInferenceResult
from .metrics_engine import metrics_engine, RealTimeMetrics


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


class OpportunityType(Enum):
    """Types of trading opportunities."""
    MOMENTUM_LONG = "momentum_long"
    MOMENTUM_SHORT = "momentum_short"
    MEAN_REVERSION_LONG = "mean_reversion_long"
    MEAN_REVERSION_SHORT = "mean_reversion_short"
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    ARBITRAGE = "arbitrage"
    NONE = "none"


@dataclass
class MarketClassification:
    """Market classification result."""
    symbol: str
    timestamp: datetime
    primary_regime: MarketRegime
    regime_confidence: float
    regime_probabilities: Dict[MarketRegime, float]
    regime_stability: float  # How stable the regime is
    transition_probability: float  # Probability of regime change
    
    # Market characteristics
    volatility_percentile: float  # 0-100
    trend_strength: float  # -1 to 1
    momentum_score: float  # -1 to 1
    mean_reversion_tendency: float  # 0-1
    
    # Clustering information
    cluster_id: int
    cluster_stability: float
    outlier_score: float
    
    # Processing metadata
    classification_confidence: float
    processing_time_ms: float


@dataclass
class TradingOpportunity:
    """Trading opportunity identification."""
    symbol: str
    timestamp: datetime
    opportunity_type: OpportunityType
    opportunity_score: float  # 0-1, higher is better
    confidence_level: float  # 0-1
    
    # Trade parameters
    suggested_action: str  # 'buy', 'sell', 'hold'
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_recommendation: float  # 0-1 fraction of portfolio
    
    # Risk metrics
    risk_reward_ratio: float
    maximum_drawdown_estimate: float
    win_probability: float
    
    # Supporting evidence
    technical_signals: List[str]
    market_conditions: List[str]
    bayesian_evidence: Dict[str, float]
    
    # Timing
    optimal_entry_window: timedelta
    expected_duration: timedelta
    urgency_score: float  # 0-1, how urgent the opportunity is


@dataclass
class MarketOpportunityAnalysis:
    """Complete market analysis combining classification and opportunities."""
    symbol: str
    timestamp: datetime
    market_classification: MarketClassification
    identified_opportunities: List[TradingOpportunity]
    overall_market_score: float  # 0-1, overall attractiveness
    risk_adjusted_score: float
    recommendation: str  # 'aggressive', 'moderate', 'conservative', 'avoid'
    analysis_confidence: float
    processing_time_ms: float


class MarketRegimeClassifier:
    """Classifier for market regimes using clustering and ML."""
    
    def __init__(self):
        self.regime_history = []
        self.regime_transition_matrix = self._initialize_transition_matrix()
        self.feature_importance_cache = {}
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize regime transition probability matrix."""
        regimes = list(MarketRegime)
        transition_matrix = {}
        
        # Initialize with uniform probabilities, will be updated with data
        for regime_from in regimes:
            transition_matrix[regime_from] = {}
            for regime_to in regimes:
                if regime_from == regime_to:
                    transition_matrix[regime_from][regime_to] = 0.7  # Higher probability to stay in same regime
                else:
                    transition_matrix[regime_from][regime_to] = 0.3 / (len(regimes) - 1)
        
        return transition_matrix
    
    async def classify_market_regime(self, market_data: Dict[str, Any], 
                                   cluster_result: ClusterResult) -> MarketClassification:
        """Classify market regime using clustering results and market features."""
        start_time = time.time()
        
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Extract market features for classification
            features = self._extract_regime_features(market_data)
            
            # Get cluster-based regime classification
            cluster_regime = self._classify_from_cluster(cluster_result, features)
            
            # Get feature-based regime probabilities
            regime_probs = self._calculate_regime_probabilities(features)
            
            # Apply Bayesian updating with historical transitions
            updated_probs = self._apply_transition_priors(regime_probs)
            
            # Determine primary regime
            primary_regime = max(updated_probs, key=updated_probs.get)
            regime_confidence = updated_probs[primary_regime]
            
            # Calculate regime stability and transition probability
            regime_stability = self._calculate_regime_stability(features, primary_regime)
            transition_probability = self._calculate_transition_probability(primary_regime, features)
            
            # Calculate market characteristics
            volatility_percentile = self._calculate_volatility_percentile(features)
            trend_strength = features.get('trend_strength', 0.0)
            momentum_score = features.get('momentum_score', 0.0)
            mean_reversion_tendency = features.get('mean_reversion_tendency', 0.5)
            
            processing_time = (time.time() - start_time) * 1000
            
            classification = MarketClassification(
                symbol=symbol,
                timestamp=datetime.now(),
                primary_regime=primary_regime,
                regime_confidence=regime_confidence,
                regime_probabilities=updated_probs,
                regime_stability=regime_stability,
                transition_probability=transition_probability,
                volatility_percentile=volatility_percentile,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                mean_reversion_tendency=mean_reversion_tendency,
                cluster_id=cluster_result.cluster_labels[0] if len(cluster_result.cluster_labels) > 0 else -1,
                cluster_stability=1.0 - cluster_result.outlier_scores[0] if len(cluster_result.outlier_scores) > 0 else 0.0,
                outlier_score=cluster_result.outlier_scores[0] if len(cluster_result.outlier_scores) > 0 else 1.0,
                classification_confidence=regime_confidence,
                processing_time_ms=processing_time
            )
            
            # Store in history for transition matrix updates
            self.regime_history.append((symbol, datetime.now(), primary_regime))
            self._update_transition_matrix()
            
            return classification
            
        except Exception as e:
            await handle_error_async(e, {'component': 'market_regime_classifier', 'symbol': market_data.get('symbol')})
            return self._create_default_classification(market_data.get('symbol', 'UNKNOWN'), start_time)
    
    def _extract_regime_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features relevant to regime classification."""
        features = {}
        
        try:
            # Price-based features
            final_signal = market_data.get('final_signal', {})
            features['price_momentum'] = final_signal.get('confidence_score', 50) / 100.0
            
            # Technical analysis features
            tech_analysis = market_data.get('technical_analysis', {})
            
            # RSI-based trend strength
            rsi = tech_analysis.get('rsi', {}).get('value', 50)
            features['rsi_normalized'] = (rsi - 50) / 50  # -1 to 1
            features['trend_strength'] = min(1.0, abs(rsi - 50) / 30)  # 0 to 1
            
            # MACD momentum
            macd_signal = tech_analysis.get('macd', {}).get('signal', 'NEUTRAL')
            features['momentum_score'] = 1.0 if 'BUY' in macd_signal else (-1.0 if 'SELL' in macd_signal else 0.0)
            
            # Bollinger Bands for volatility regime
            bb_data = tech_analysis.get('bollinger', {})
            bb_position = bb_data.get('position', 0.5)  # 0-1, position within bands
            features['volatility_regime'] = abs(bb_position - 0.5) * 2  # 0-1, distance from center
            
            # Volume analysis
            volume_data = market_data.get('volume_analysis', {})
            features['volume_momentum'] = volume_data.get('volume_trend', 0.0)
            
            # Market data features
            market = market_data.get('market_data', {})
            current_price = market.get('current_price', 1)
            volatility = market.get('volatility', 0.05)
            
            features['volatility'] = volatility
            features['volatility_normalized'] = min(1.0, volatility / 0.1)  # Normalize to 0-1
            
            # Mean reversion tendency (based on price position relative to moving averages)
            sma_20 = tech_analysis.get('sma_20', {}).get('value', current_price)
            if sma_20 > 0:
                price_to_sma = (current_price - sma_20) / sma_20
                features['mean_reversion_tendency'] = 1.0 / (1.0 + abs(price_to_sma))  # 0-1, higher = more mean reverting
            else:
                features['mean_reversion_tendency'] = 0.5
            
            # Breakout potential (based on volatility and price action)
            atr = tech_analysis.get('atr', {}).get('value', volatility * current_price)
            price_range = market.get('high_24h', current_price) - market.get('low_24h', current_price)
            if atr > 0:
                features['breakout_potential'] = min(1.0, price_range / atr)
            else:
                features['breakout_potential'] = 0.5
            
            return features
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to extract regime features: {e}")
            return {
                'price_momentum': 0.0,
                'trend_strength': 0.0,
                'momentum_score': 0.0,
                'volatility_regime': 0.5,
                'volume_momentum': 0.0,
                'volatility': 0.05,
                'mean_reversion_tendency': 0.5,
                'breakout_potential': 0.5
            }
    
    def _classify_from_cluster(self, cluster_result: ClusterResult, features: Dict[str, float]) -> MarketRegime:
        """Classify regime based on cluster characteristics."""
        try:
            if cluster_result.n_clusters == 0:
                return MarketRegime.UNKNOWN
            
            # Use cluster characteristics to determine regime
            cluster_id = cluster_result.cluster_labels[0] if len(cluster_result.cluster_labels) > 0 else -1
            
            if cluster_id == -1:  # Outlier
                # High volatility regime for outliers
                return MarketRegime.HIGH_VOLATILITY if features.get('volatility_normalized', 0) > 0.7 else MarketRegime.UNKNOWN
            
            # Determine regime based on cluster center characteristics
            # This is a simplified heuristic - in practice, you'd train this mapping
            volatility = features.get('volatility_normalized', 0.5)
            trend_strength = features.get('trend_strength', 0.0)
            momentum = features.get('momentum_score', 0.0)
            
            if volatility > 0.7:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.3:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.6:
                if momentum > 0.3:
                    return MarketRegime.BULL_TRENDING
                elif momentum < -0.3:
                    return MarketRegime.BEAR_TRENDING
                else:
                    return MarketRegime.SIDEWAYS_CONSOLIDATION
            elif features.get('breakout_potential', 0) > 0.7:
                return MarketRegime.BREAKOUT
            else:
                return MarketRegime.SIDEWAYS_CONSOLIDATION
                
        except Exception as e:
            crypto_logger.logger.warning(f"Cluster-based classification failed: {e}")
            return MarketRegime.UNKNOWN
    
    def _calculate_regime_probabilities(self, features: Dict[str, float]) -> Dict[MarketRegime, float]:
        """Calculate probabilities for each regime based on features."""
        try:
            probs = {}
            
            volatility = features.get('volatility_normalized', 0.5)
            trend_strength = features.get('trend_strength', 0.0)
            momentum = features.get('momentum_score', 0.0)
            mean_reversion = features.get('mean_reversion_tendency', 0.5)
            breakout_potential = features.get('breakout_potential', 0.5)
            
            # High/Low volatility regimes
            probs[MarketRegime.HIGH_VOLATILITY] = max(0.1, volatility)
            probs[MarketRegime.LOW_VOLATILITY] = max(0.1, 1 - volatility)
            
            # Trending regimes
            trend_prob = trend_strength * 0.8
            if momentum > 0:
                probs[MarketRegime.BULL_TRENDING] = trend_prob * (0.5 + momentum * 0.5)
                probs[MarketRegime.BEAR_TRENDING] = trend_prob * (0.5 - momentum * 0.5)
            else:
                probs[MarketRegime.BULL_TRENDING] = trend_prob * (0.5 + momentum * 0.5)
                probs[MarketRegime.BEAR_TRENDING] = trend_prob * (0.5 - momentum * 0.5)
            
            # Sideways/consolidation
            probs[MarketRegime.SIDEWAYS_CONSOLIDATION] = (1 - trend_strength) * mean_reversion
            
            # Breakout regime
            probs[MarketRegime.BREAKOUT] = breakout_potential * volatility
            
            # Reversal (based on extreme momentum with high mean reversion)
            probs[MarketRegime.REVERSAL] = abs(momentum) * mean_reversion
            
            # Accumulation/Distribution (simplified)
            volume_momentum = features.get('volume_momentum', 0.0)
            probs[MarketRegime.ACCUMULATION] = max(0.1, volume_momentum) * (1 - volatility)
            probs[MarketRegime.DISTRIBUTION] = max(0.1, -volume_momentum) * (1 - volatility)
            
            # Unknown (baseline)
            probs[MarketRegime.UNKNOWN] = 0.1
            
            # Normalize probabilities
            total = sum(probs.values())
            if total > 0:
                for regime in probs:
                    probs[regime] /= total
            
            return probs
            
        except Exception as e:
            crypto_logger.logger.warning(f"Regime probability calculation failed: {e}")
            # Return uniform probabilities
            regimes = list(MarketRegime)
            return {regime: 1.0 / len(regimes) for regime in regimes}
    
    def _apply_transition_priors(self, regime_probs: Dict[MarketRegime, float]) -> Dict[MarketRegime, float]:
        """Apply transition probability priors based on recent regime history."""
        try:
            if not self.regime_history:
                return regime_probs
            
            # Get recent regime
            recent_regime = self.regime_history[-1][2]
            
            # Apply transition matrix
            updated_probs = {}
            for regime in regime_probs:
                # Bayesian update: P(regime | data) ∝ P(data | regime) * P(regime | previous)
                likelihood = regime_probs[regime]
                prior = self.regime_transition_matrix[recent_regime].get(regime, 0.1)
                updated_probs[regime] = likelihood * prior
            
            # Normalize
            total = sum(updated_probs.values())
            if total > 0:
                for regime in updated_probs:
                    updated_probs[regime] /= total
            
            return updated_probs
            
        except Exception as e:
            crypto_logger.logger.warning(f"Transition prior application failed: {e}")
            return regime_probs
    
    def _calculate_regime_stability(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate stability score for the current regime."""
        try:
            # Stability based on feature consistency with regime
            volatility = features.get('volatility_normalized', 0.5)
            trend_strength = features.get('trend_strength', 0.0)
            
            if regime == MarketRegime.HIGH_VOLATILITY:
                return volatility  # High volatility = stable high vol regime
            elif regime == MarketRegime.LOW_VOLATILITY:
                return 1 - volatility
            elif regime in [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]:
                return trend_strength  # Strong trend = stable trending regime
            elif regime == MarketRegime.SIDEWAYS_CONSOLIDATION:
                return 1 - trend_strength  # Weak trend = stable sideways
            else:
                return 0.5  # Default stability
                
        except Exception:
            return 0.5
    
    def _calculate_transition_probability(self, current_regime: MarketRegime, features: Dict[str, float]) -> float:
        """Calculate probability of regime transition."""
        try:
            # Higher volatility and extreme values increase transition probability
            volatility = features.get('volatility_normalized', 0.5)
            trend_strength = features.get('trend_strength', 0.0)
            momentum = abs(features.get('momentum_score', 0.0))
            
            # Base transition probability
            base_transition = 0.1
            
            # Volatility increases transition probability
            volatility_effect = volatility * 0.3
            
            # Extreme momentum increases transition probability
            momentum_effect = momentum * 0.2
            
            # Very strong trends are less likely to transition
            trend_stability = max(0, 1 - trend_strength) * 0.2
            
            transition_prob = base_transition + volatility_effect + momentum_effect + trend_stability
            
            return min(0.8, transition_prob)  # Cap at 80%
            
        except Exception:
            return 0.3  # Default transition probability
    
    def _calculate_volatility_percentile(self, features: Dict[str, float]) -> float:
        """Calculate volatility percentile (0-100)."""
        volatility = features.get('volatility_normalized', 0.5)
        return volatility * 100
    
    def _update_transition_matrix(self):
        """Update transition matrix based on observed regime changes."""
        try:
            if len(self.regime_history) < 2:
                return
            
            # Count transitions in recent history
            recent_history = self.regime_history[-100:]  # Last 100 observations
            transition_counts = {}
            
            for i in range(1, len(recent_history)):
                regime_from = recent_history[i-1][2]
                regime_to = recent_history[i][2]
                
                if regime_from not in transition_counts:
                    transition_counts[regime_from] = {}
                
                transition_counts[regime_from][regime_to] = transition_counts[regime_from].get(regime_to, 0) + 1
            
            # Update transition matrix with smoothing
            smoothing_factor = 0.1
            
            for regime_from in transition_counts:
                total_from = sum(transition_counts[regime_from].values())
                
                for regime_to in transition_counts[regime_from]:
                    observed_prob = transition_counts[regime_from][regime_to] / total_from
                    current_prob = self.regime_transition_matrix[regime_from].get(regime_to, 0.1)
                    
                    # Exponential moving average update
                    updated_prob = (1 - smoothing_factor) * current_prob + smoothing_factor * observed_prob
                    self.regime_transition_matrix[regime_from][regime_to] = updated_prob
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to update transition matrix: {e}")
    
    def _create_default_classification(self, symbol: str, start_time: float) -> MarketClassification:
        """Create default classification for error cases."""
        return MarketClassification(
            symbol=symbol,
            timestamp=datetime.now(),
            primary_regime=MarketRegime.UNKNOWN,
            regime_confidence=0.1,
            regime_probabilities={regime: 1.0/len(MarketRegime) for regime in MarketRegime},
            regime_stability=0.5,
            transition_probability=0.5,
            volatility_percentile=50.0,
            trend_strength=0.0,
            momentum_score=0.0,
            mean_reversion_tendency=0.5,
            cluster_id=-1,
            cluster_stability=0.0,
            outlier_score=1.0,
            classification_confidence=0.1,
            processing_time_ms=(time.time() - start_time) * 1000
        )


class OpportunityDetector:
    """Detector for trading opportunities based on market classification."""
    
    def __init__(self):
        self.opportunity_history = []
        self.success_rates = {}  # Track success rates by opportunity type
    
    async def identify_opportunities(self, market_classification: MarketClassification,
                                   market_data: Dict[str, Any],
                                   bayesian_result: BayesianInferenceResult) -> List[TradingOpportunity]:
        """Identify trading opportunities based on market classification and analysis."""
        try:
            opportunities = []
            
            # Momentum opportunities
            momentum_ops = self._detect_momentum_opportunities(market_classification, market_data)
            opportunities.extend(momentum_ops)
            
            # Mean reversion opportunities
            reversion_ops = self._detect_mean_reversion_opportunities(market_classification, market_data)
            opportunities.extend(reversion_ops)
            
            # Breakout opportunities
            breakout_ops = self._detect_breakout_opportunities(market_classification, market_data)
            opportunities.extend(breakout_ops)
            
            # Volatility opportunities
            volatility_ops = self._detect_volatility_opportunities(market_classification, market_data)
            opportunities.extend(volatility_ops)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_and_rank_opportunities(
                opportunities, bayesian_result
            )
            
            # Store in history
            for opp in filtered_opportunities:
                self.opportunity_history.append((datetime.now(), opp))
            
            # Limit history
            if len(self.opportunity_history) > 1000:
                self.opportunity_history = self.opportunity_history[-1000:]
            
            return filtered_opportunities
            
        except Exception as e:
            await handle_error_async(e, {'component': 'opportunity_detector'})
            return []
    
    def _detect_momentum_opportunities(self, classification: MarketClassification,
                                     market_data: Dict[str, Any]) -> List[TradingOpportunity]:
        """Detect momentum-based trading opportunities."""
        opportunities = []
        
        try:
            if classification.primary_regime in [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]:
                # Strong momentum opportunity
                is_bullish = classification.momentum_score > 0
                trend_strength = classification.trend_strength
                
                if trend_strength > 0.6:  # Strong trend
                    opp_type = OpportunityType.MOMENTUM_LONG if is_bullish else OpportunityType.MOMENTUM_SHORT
                    
                    opportunity_score = trend_strength * classification.regime_confidence
                    
                    # Calculate trade parameters
                    current_price = market_data.get('market_data', {}).get('current_price', 0)
                    volatility = classification.volatility_percentile / 100
                    
                    if current_price > 0:
                        stop_loss_pct = 0.02 + volatility * 0.03  # 2-5% stop loss
                        take_profit_pct = stop_loss_pct * 2  # 2:1 risk-reward
                        
                        if is_bullish:
                            stop_loss = current_price * (1 - stop_loss_pct)
                            take_profit = current_price * (1 + take_profit_pct)
                        else:
                            stop_loss = current_price * (1 + stop_loss_pct)
                            take_profit = current_price * (1 - take_profit_pct)
                        
                        opportunity = TradingOpportunity(
                            symbol=classification.symbol,
                            timestamp=datetime.now(),
                            opportunity_type=opp_type,
                            opportunity_score=opportunity_score,
                            confidence_level=classification.regime_confidence,
                            suggested_action='buy' if is_bullish else 'sell',
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            position_size_recommendation=min(0.1, opportunity_score),
                            risk_reward_ratio=2.0,
                            maximum_drawdown_estimate=stop_loss_pct,
                            win_probability=0.6 + trend_strength * 0.2,
                            technical_signals=[f"Strong {classification.primary_regime.value}"],
                            market_conditions=[f"Trend strength: {trend_strength:.2f}"],
                            bayesian_evidence={},
                            optimal_entry_window=timedelta(hours=1),
                            expected_duration=timedelta(days=3),
                            urgency_score=trend_strength
                        )
                        
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            crypto_logger.logger.warning(f"Momentum opportunity detection failed: {e}")
            return []
    
    def _detect_mean_reversion_opportunities(self, classification: MarketClassification,
                                           market_data: Dict[str, Any]) -> List[TradingOpportunity]:
        """Detect mean reversion opportunities."""
        opportunities = []
        
        try:
            if classification.mean_reversion_tendency > 0.7 and classification.primary_regime == MarketRegime.SIDEWAYS_CONSOLIDATION:
                # Strong mean reversion tendency
                current_price = market_data.get('market_data', {}).get('current_price', 0)
                
                if current_price > 0:
                    # Determine if price is extended from mean
                    tech_analysis = market_data.get('technical_analysis', {})
                    rsi = tech_analysis.get('rsi', {}).get('value', 50)
                    
                    if rsi > 70:  # Overbought - short opportunity
                        opp_type = OpportunityType.MEAN_REVERSION_SHORT
                        action = 'sell'
                        entry_bias = -0.02  # Expect 2% decline
                    elif rsi < 30:  # Oversold - long opportunity
                        opp_type = OpportunityType.MEAN_REVERSION_LONG
                        action = 'buy'
                        entry_bias = 0.02  # Expect 2% rise
                    else:
                        return opportunities
                    
                    opportunity_score = classification.mean_reversion_tendency * (abs(rsi - 50) / 50)
                    
                    stop_loss_pct = 0.015  # Tight stops for mean reversion
                    take_profit_pct = 0.03
                    
                    if action == 'buy':
                        stop_loss = current_price * (1 - stop_loss_pct)
                        take_profit = current_price * (1 + take_profit_pct)
                    else:
                        stop_loss = current_price * (1 + stop_loss_pct)
                        take_profit = current_price * (1 - take_profit_pct)
                    
                    opportunity = TradingOpportunity(
                        symbol=classification.symbol,
                        timestamp=datetime.now(),
                        opportunity_type=opp_type,
                        opportunity_score=opportunity_score,
                        confidence_level=classification.mean_reversion_tendency,
                        suggested_action=action,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size_recommendation=min(0.05, opportunity_score * 0.5),
                        risk_reward_ratio=2.0,
                        maximum_drawdown_estimate=stop_loss_pct,
                        win_probability=0.7,  # Mean reversion often has higher win rate
                        technical_signals=[f"RSI {rsi:.1f}", "Mean reversion setup"],
                        market_conditions=["Sideways consolidation", f"Mean reversion tendency: {classification.mean_reversion_tendency:.2f}"],
                        bayesian_evidence={},
                        optimal_entry_window=timedelta(hours=4),
                        expected_duration=timedelta(days=1),
                        urgency_score=abs(rsi - 50) / 50
                    )
                    
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            crypto_logger.logger.warning(f"Mean reversion opportunity detection failed: {e}")
            return []
    
    def _detect_breakout_opportunities(self, classification: MarketClassification,
                                     market_data: Dict[str, Any]) -> List[TradingOpportunity]:
        """Detect breakout opportunities."""
        opportunities = []
        
        try:
            if classification.primary_regime == MarketRegime.BREAKOUT:
                current_price = market_data.get('market_data', {}).get('current_price', 0)
                
                if current_price > 0:
                    # Determine breakout direction from momentum
                    is_bullish_breakout = classification.momentum_score > 0
                    
                    opp_type = OpportunityType.BREAKOUT_LONG if is_bullish_breakout else OpportunityType.BREAKOUT_SHORT
                    
                    opportunity_score = classification.regime_confidence
                    
                    # Breakouts often have wider stops
                    stop_loss_pct = 0.03 + classification.volatility_percentile / 100 * 0.02
                    take_profit_pct = stop_loss_pct * 3  # 3:1 risk-reward for breakouts
                    
                    if is_bullish_breakout:
                        action = 'buy'
                        stop_loss = current_price * (1 - stop_loss_pct)
                        take_profit = current_price * (1 + take_profit_pct)
                    else:
                        action = 'sell'
                        stop_loss = current_price * (1 + stop_loss_pct)
                        take_profit = current_price * (1 - take_profit_pct)
                    
                    opportunity = TradingOpportunity(
                        symbol=classification.symbol,
                        timestamp=datetime.now(),
                        opportunity_type=opp_type,
                        opportunity_score=opportunity_score,
                        confidence_level=classification.regime_confidence,
                        suggested_action=action,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size_recommendation=min(0.08, opportunity_score),
                        risk_reward_ratio=3.0,
                        maximum_drawdown_estimate=stop_loss_pct,
                        win_probability=0.45,  # Breakouts have lower win rate but higher reward
                        technical_signals=["Breakout pattern", f"Momentum: {classification.momentum_score:.2f}"],
                        market_conditions=[f"Volatility: {classification.volatility_percentile:.0f}th percentile"],
                        bayesian_evidence={},
                        optimal_entry_window=timedelta(minutes=30),
                        expected_duration=timedelta(hours=12),
                        urgency_score=0.8  # Breakouts are time-sensitive
                    )
                    
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            crypto_logger.logger.warning(f"Breakout opportunity detection failed: {e}")
            return []
    
    def _detect_volatility_opportunities(self, classification: MarketClassification,
                                       market_data: Dict[str, Any]) -> List[TradingOpportunity]:
        """Detect volatility-based opportunities."""
        opportunities = []
        
        try:
            volatility_percentile = classification.volatility_percentile
            
            if volatility_percentile > 80:  # High volatility - expect contraction
                opp_type = OpportunityType.VOLATILITY_CONTRACTION
                opportunity_score = (volatility_percentile - 80) / 20  # 0-1 scale
                
                # Volatility contraction strategies (simplified)
                opportunity = TradingOpportunity(
                    symbol=classification.symbol,
                    timestamp=datetime.now(),
                    opportunity_type=opp_type,
                    opportunity_score=opportunity_score,
                    confidence_level=0.6,
                    suggested_action='hold',  # Complex volatility strategies
                    entry_price=market_data.get('market_data', {}).get('current_price'),
                    stop_loss=None,
                    take_profit=None,
                    position_size_recommendation=0.02,
                    risk_reward_ratio=1.5,
                    maximum_drawdown_estimate=0.05,
                    win_probability=0.6,
                    technical_signals=[f"High volatility: {volatility_percentile:.0f}th percentile"],
                    market_conditions=["Expect volatility contraction"],
                    bayesian_evidence={},
                    optimal_entry_window=timedelta(hours=6),
                    expected_duration=timedelta(days=2),
                    urgency_score=0.4
                )
                
                opportunities.append(opportunity)
            
            elif volatility_percentile < 20:  # Low volatility - expect expansion
                opp_type = OpportunityType.VOLATILITY_EXPANSION
                opportunity_score = (20 - volatility_percentile) / 20
                
                opportunity = TradingOpportunity(
                    symbol=classification.symbol,
                    timestamp=datetime.now(),
                    opportunity_type=opp_type,
                    opportunity_score=opportunity_score,
                    confidence_level=0.5,
                    suggested_action='hold',
                    entry_price=market_data.get('market_data', {}).get('current_price'),
                    stop_loss=None,
                    take_profit=None,
                    position_size_recommendation=0.03,
                    risk_reward_ratio=2.0,
                    maximum_drawdown_estimate=0.03,
                    win_probability=0.5,
                    technical_signals=[f"Low volatility: {volatility_percentile:.0f}th percentile"],
                    market_conditions=["Expect volatility expansion"],
                    bayesian_evidence={},
                    optimal_entry_window=timedelta(hours=12),
                    expected_duration=timedelta(days=3),
                    urgency_score=0.3
                )
                
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            crypto_logger.logger.warning(f"Volatility opportunity detection failed: {e}")
            return []
    
    def _filter_and_rank_opportunities(self, opportunities: List[TradingOpportunity],
                                     bayesian_result: BayesianInferenceResult) -> List[TradingOpportunity]:
        """Filter and rank opportunities by quality and consistency with Bayesian analysis."""
        try:
            if not opportunities:
                return []
            
            # Filter by minimum opportunity score
            min_score = 0.3
            filtered = [opp for opp in opportunities if opp.opportunity_score >= min_score]
            
            # Adjust scores based on Bayesian analysis consistency
            for opp in filtered:
                bayesian_consistency = self._calculate_bayesian_consistency(opp, bayesian_result)
                opp.opportunity_score *= bayesian_consistency
                opp.confidence_level *= bayesian_consistency
                
                # Store Bayesian evidence
                opp.bayesian_evidence = {
                    'bayesian_decision': bayesian_result.decision,
                    'bayesian_confidence': bayesian_result.confidence_score,
                    'consistency_score': bayesian_consistency
                }
            
            # Sort by adjusted opportunity score
            filtered.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Return top 5 opportunities
            return filtered[:5]
            
        except Exception as e:
            crypto_logger.logger.warning(f"Opportunity filtering failed: {e}")
            return opportunities[:3]  # Return first 3 as fallback
    
    def _calculate_bayesian_consistency(self, opportunity: TradingOpportunity,
                                      bayesian_result: BayesianInferenceResult) -> float:
        """Calculate consistency between opportunity and Bayesian analysis."""
        try:
            # Check if opportunity direction matches Bayesian decision
            opp_direction = opportunity.suggested_action
            bayesian_decision = bayesian_result.decision
            
            consistency_score = 0.5  # Base score
            
            # Direction consistency
            if (opp_direction == 'buy' and bayesian_decision in ['buy', 'strong_buy']) or \
               (opp_direction == 'sell' and bayesian_decision in ['sell', 'strong_sell']):
                consistency_score += 0.3
            elif opp_direction == 'hold' or bayesian_decision == 'hold':
                consistency_score += 0.1
            else:
                consistency_score -= 0.2
            
            # Confidence alignment
            confidence_alignment = min(opportunity.confidence_level, bayesian_result.confidence_score)
            consistency_score += confidence_alignment * 0.2
            
            return max(0.1, min(1.0, consistency_score))
            
        except Exception:
            return 0.7  # Default consistency score


class AdvancedMarketClassificationEngine:
    """Main engine combining market classification and opportunity identification."""
    
    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()
        self.opportunity_detector = OpportunityDetector()
        self.analysis_history = []
    
    async def analyze_market_comprehensively(self, market_data: Dict[str, Any]) -> MarketOpportunityAnalysis:
        """Perform comprehensive market analysis with classification and opportunity identification."""
        start_time = time.time()
        
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Step 1: Feature extraction and clustering
            feature_result = await feature_pipeline.process_market_data(
                pd.DataFrame([market_data.get('market_data', {})]),
                target_column=None
            )
            
            # Step 2: Clustering analysis
            if feature_result.processed_data.size > 0:
                cluster_result = await clustering_engine.cluster_market_data(
                    pd.DataFrame(feature_result.processed_data, columns=feature_result.feature_names)
                )
            else:
                # Create empty cluster result
                from .clustering_engine import ClusterResult
                cluster_result = ClusterResult(
                    cluster_labels=np.array([-1]),
                    cluster_probabilities=np.array([0.0]),
                    n_clusters=0,
                    outlier_scores=np.array([1.0]),
                    silhouette_score=0.0,
                    processing_time_ms=0.0,
                    feature_importance={},
                    cluster_centers=np.array([]),
                    cluster_sizes={},
                    validity_metrics={}
                )
            
            # Step 3: Market regime classification
            market_classification = await self.regime_classifier.classify_market_regime(
                market_data, cluster_result
            )
            
            # Step 4: Bayesian analysis
            bayesian_result = await bayesian_framework.analyze_market_bayesian(market_data)
            
            # Step 5: Opportunity identification
            opportunities = await self.opportunity_detector.identify_opportunities(
                market_classification, market_data, bayesian_result
            )
            
            # Step 6: Overall market assessment
            overall_score = self._calculate_overall_market_score(
                market_classification, opportunities, bayesian_result
            )
            
            risk_adjusted_score = self._calculate_risk_adjusted_score(
                overall_score, market_classification, bayesian_result
            )
            
            recommendation = self._generate_recommendation(risk_adjusted_score, market_classification)
            
            analysis_confidence = self._calculate_analysis_confidence(
                market_classification, bayesian_result, opportunities
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            analysis = MarketOpportunityAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                market_classification=market_classification,
                identified_opportunities=opportunities,
                overall_market_score=overall_score,
                risk_adjusted_score=risk_adjusted_score,
                recommendation=recommendation,
                analysis_confidence=analysis_confidence,
                processing_time_ms=processing_time
            )
            
            # Store in history
            self.analysis_history.append(analysis)
            if len(self.analysis_history) > 500:
                self.analysis_history = self.analysis_history[-500:]
            
            crypto_logger.logger.info(
                f"Market analysis for {symbol}: {market_classification.primary_regime.value} "
                f"({len(opportunities)} opportunities, score: {overall_score:.2f})"
            )
            
            return analysis
            
        except Exception as e:
            await handle_error_async(e, {'component': 'market_classification_engine'})
            return self._create_default_analysis(market_data.get('symbol', 'UNKNOWN'), start_time)
    
    def _calculate_overall_market_score(self, classification: MarketClassification,
                                      opportunities: List[TradingOpportunity],
                                      bayesian_result: BayesianInferenceResult) -> float:
        """Calculate overall market attractiveness score."""
        try:
            # Base score from regime confidence
            base_score = classification.regime_confidence
            
            # Opportunity quality boost
            if opportunities:
                avg_opportunity_score = np.mean([opp.opportunity_score for opp in opportunities])
                opportunity_boost = avg_opportunity_score * 0.3
            else:
                opportunity_boost = 0.0
            
            # Bayesian confidence boost
            bayesian_boost = bayesian_result.confidence_score * 0.2
            
            # Stability penalty (unstable regimes are riskier)
            stability_penalty = (1 - classification.regime_stability) * 0.1
            
            overall_score = base_score + opportunity_boost + bayesian_boost - stability_penalty
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception:
            return 0.5
    
    def _calculate_risk_adjusted_score(self, overall_score: float,
                                     classification: MarketClassification,
                                     bayesian_result: BayesianInferenceResult) -> float:
        """Calculate risk-adjusted market score."""
        try:
            # Volatility penalty
            volatility_penalty = classification.volatility_percentile / 100 * 0.2
            
            # Uncertainty penalty from Bayesian analysis
            uncertainty_penalty = bayesian_result.uncertainty_score * 0.15
            
            # Regime transition risk
            transition_penalty = classification.transition_probability * 0.1
            
            total_penalty = volatility_penalty + uncertainty_penalty + transition_penalty
            
            risk_adjusted = overall_score * (1 - total_penalty)
            
            return max(0.0, risk_adjusted)
            
        except Exception:
            return overall_score * 0.7  # Conservative adjustment
    
    def _generate_recommendation(self, risk_adjusted_score: float,
                               classification: MarketClassification) -> str:
        """Generate trading recommendation based on analysis."""
        try:
            if risk_adjusted_score > 0.8 and classification.regime_stability > 0.7:
                return 'aggressive'
            elif risk_adjusted_score > 0.6:
                return 'moderate'
            elif risk_adjusted_score > 0.4:
                return 'conservative'
            else:
                return 'avoid'
                
        except Exception:
            return 'conservative'
    
    def _calculate_analysis_confidence(self, classification: MarketClassification,
                                     bayesian_result: BayesianInferenceResult,
                                     opportunities: List[TradingOpportunity]) -> float:
        """Calculate confidence in the overall analysis."""
        try:
            # Average of component confidences
            classification_confidence = classification.classification_confidence
            bayesian_confidence = bayesian_result.confidence_score
            
            if opportunities:
                opportunity_confidence = np.mean([opp.confidence_level for opp in opportunities])
            else:
                opportunity_confidence = 0.3  # Low confidence if no opportunities
            
            overall_confidence = np.mean([
                classification_confidence,
                bayesian_confidence,
                opportunity_confidence
            ])
            
            return overall_confidence
            
        except Exception:
            return 0.5
    
    def _create_default_analysis(self, symbol: str, start_time: float) -> MarketOpportunityAnalysis:
        """Create default analysis for error cases."""
        default_classification = self.regime_classifier._create_default_classification(symbol, start_time)
        
        return MarketOpportunityAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            market_classification=default_classification,
            identified_opportunities=[],
            overall_market_score=0.3,
            risk_adjusted_score=0.2,
            recommendation='avoid',
            analysis_confidence=0.1,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        if not self.analysis_history:
            return {}
        
        recent_analyses = self.analysis_history[-50:]
        
        processing_times = [a.processing_time_ms for a in recent_analyses]
        overall_scores = [a.overall_market_score for a in recent_analyses]
        
        regime_distribution = {}
        for analysis in recent_analyses:
            regime = analysis.market_classification.primary_regime.value
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        
        return {
            'total_analyses': len(self.analysis_history),
            'avg_processing_time_ms': np.mean(processing_times),
            'avg_market_score': np.mean(overall_scores),
            'regime_distribution': regime_distribution,
            'avg_opportunities_per_analysis': np.mean([len(a.identified_opportunities) for a in recent_analyses]),
            'recommendation_distribution': {
                rec: sum(1 for a in recent_analyses if a.recommendation == rec)
                for rec in ['aggressive', 'moderate', 'conservative', 'avoid']
            }
        }


# Global market classification engine instance
market_classification_engine = AdvancedMarketClassificationEngine()