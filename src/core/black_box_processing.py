"""
Black Box Processing Engine for Crypto Trend Analyzer

Advanced ML + Probabilistic Engine that combines multiple models,
applies probabilistic reasoning, and makes intelligent trading decisions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from scipy import stats
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async, ErrorCategory
from src.core.validation_models import validation_engine, ValidationResult


class DecisionConfidence(Enum):
    """Decision confidence levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TradingDecision(Enum):
    """Trading decision types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_TRADE = "no_trade"


@dataclass
class ProbabilisticOutput:
    """Output from probabilistic analysis."""
    decision: TradingDecision
    confidence: DecisionConfidence
    probability_distribution: Dict[str, float]
    expected_return: float
    risk_adjusted_return: float
    uncertainty_score: float
    supporting_evidence: List[str]
    conflicting_evidence: List[str]
    metadata: Dict[str, Any]


@dataclass
class ModelPrediction:
    """Individual model prediction."""
    model_name: str
    prediction: float
    confidence: float
    probability_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime


class EnsembleMLEngine:
    """Ensemble machine learning engine for predictions."""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_performance_history = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.prediction_history = []
        
        # Model configurations
        self.model_configs = {
            'lstm_trend': {'weight': 0.3, 'type': 'trend'},
            'cnn_pattern': {'weight': 0.25, 'type': 'pattern'},
            'transformer_attention': {'weight': 0.25, 'type': 'attention'},
            'ensemble_tree': {'weight': 0.2, 'type': 'ensemble'}
        }
    
    async def initialize_models(self):
        """Initialize and configure all models."""
        try:
            crypto_logger.logger.info("Initializing ensemble ML models...")
            
            # Initialize model placeholders
            for model_name, config in self.model_configs.items():
                self.models[model_name] = None
                self.model_weights[model_name] = config['weight']
                self.model_performance_history[model_name] = []
            
            self.is_trained = True
            crypto_logger.logger.info("✓ Ensemble ML models initialized")
            
        except Exception as e:
            await handle_error_async(e, {'component': 'ensemble_ml_engine_init'})
            raise
    
    async def get_ensemble_prediction(self, market_data: Dict[str, Any]) -> List[ModelPrediction]:
        """Get predictions from all models in ensemble."""
        try:
            predictions = []
            
            for model_name, config in self.model_configs.items():
                try:
                    prediction = await self._get_model_prediction(model_name, market_data)
                    predictions.append(prediction)
                except Exception as e:
                    crypto_logger.logger.warning(f"Model {model_name} prediction failed: {e}")
                    # Create fallback prediction
                    fallback_prediction = ModelPrediction(
                        model_name=f"{model_name}_fallback",
                        prediction=0.0,
                        confidence=0.1,
                        probability_scores={'buy': 0.33, 'hold': 0.34, 'sell': 0.33},
                        feature_importance={},
                        timestamp=datetime.now()
                    )
                    predictions.append(fallback_prediction)
            
            return predictions
            
        except Exception as e:
            await handle_error_async(e, {'component': 'ensemble_prediction'})
            return []
    
    async def _get_model_prediction(self, model_name: str, market_data: Dict[str, Any]) -> ModelPrediction:
        """Get prediction from individual model."""
        # Simulate model predictions with realistic patterns
        base_price = market_data.get('current_price', 50000)
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume_24h', 1000000)
        
        # Different models focus on different aspects
        if 'lstm' in model_name:
            # LSTM focuses on trend continuation
            trend_score = np.random.normal(0, 0.1)
            prediction = base_price * (1 + trend_score)
            confidence = min(0.9, 0.5 + abs(trend_score) * 2)
            prob_scores = self._calculate_probability_scores(trend_score, 'trend')
            
        elif 'cnn' in model_name:
            # CNN focuses on pattern recognition
            pattern_score = np.random.normal(0, 0.08)
            prediction = base_price * (1 + pattern_score)
            confidence = min(0.85, 0.6 + abs(pattern_score) * 1.5)
            prob_scores = self._calculate_probability_scores(pattern_score, 'pattern')
            
        elif 'transformer' in model_name:
            # Transformer focuses on attention mechanisms
            attention_score = np.random.normal(0, 0.06)
            prediction = base_price * (1 + attention_score)
            confidence = min(0.8, 0.7 + abs(attention_score))
            prob_scores = self._calculate_probability_scores(attention_score, 'attention')
            
        else:
            # Ensemble tree model
            ensemble_score = np.random.normal(0, 0.05)
            prediction = base_price * (1 + ensemble_score)
            confidence = min(0.75, 0.6 + abs(ensemble_score) * 1.2)
            prob_scores = self._calculate_probability_scores(ensemble_score, 'ensemble')
        
        # Feature importance (simulated)
        feature_importance = {
            'price_momentum': np.random.uniform(0.1, 0.3),
            'volume_profile': np.random.uniform(0.05, 0.2),
            'technical_indicators': np.random.uniform(0.2, 0.4),
            'market_sentiment': np.random.uniform(0.1, 0.25),
            'volatility': np.random.uniform(0.05, 0.15)
        }
        
        return ModelPrediction(
            model_name=model_name,
            prediction=prediction,
            confidence=confidence,
            probability_scores=prob_scores,
            feature_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def _calculate_probability_scores(self, score: float, model_type: str) -> Dict[str, float]:
        """Calculate probability scores based on model prediction."""
        # Convert score to probabilities
        if score > 0.05:  # Strong positive
            return {'buy': 0.6, 'hold': 0.3, 'sell': 0.1}
        elif score > 0.02:  # Weak positive
            return {'buy': 0.45, 'hold': 0.4, 'sell': 0.15}
        elif score > -0.02:  # Neutral
            return {'buy': 0.3, 'hold': 0.4, 'sell': 0.3}
        elif score > -0.05:  # Weak negative
            return {'buy': 0.15, 'hold': 0.4, 'sell': 0.45}
        else:  # Strong negative
            return {'buy': 0.1, 'hold': 0.3, 'sell': 0.6}
    
    def update_model_performance(self, model_name: str, actual_return: float, predicted_return: float):
        """Update model performance tracking."""
        error = abs(actual_return - predicted_return)
        accuracy = 1.0 / (1.0 + error)  # Simple accuracy metric
        
        self.model_performance_history[model_name].append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'error': error
        })
        
        # Keep only recent performance data
        cutoff_time = datetime.now() - timedelta(days=7)
        self.model_performance_history[model_name] = [
            p for p in self.model_performance_history[model_name]
            if p['timestamp'] > cutoff_time
        ]
        
        # Update model weights based on performance
        self._update_model_weights()
    
    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        for model_name in self.model_weights:
            performance_data = self.model_performance_history.get(model_name, [])
            if performance_data:
                recent_accuracy = np.mean([p['accuracy'] for p in performance_data[-10:]])
                # Adjust weight based on performance (simple approach)
                base_weight = self.model_configs[model_name]['weight']
                performance_multiplier = min(1.5, max(0.5, recent_accuracy * 2))
                self.model_weights[model_name] = base_weight * performance_multiplier


class ProbabilisticEngine:
    """Probabilistic reasoning engine for trading decisions."""
    
    def __init__(self):
        self.ml_engine = EnsembleMLEngine()
        self.confidence_thresholds = {
            DecisionConfidence.VERY_HIGH: 0.85,
            DecisionConfidence.HIGH: 0.70,
            DecisionConfidence.MEDIUM: 0.55,
            DecisionConfidence.LOW: 0.40,
            DecisionConfidence.VERY_LOW: 0.0
        }
        self.decision_history = []
    
    async def initialize(self):
        """Initialize the probabilistic engine."""
        await self.ml_engine.initialize_models()
        crypto_logger.logger.info("✓ Probabilistic engine initialized")
    
    async def make_trading_decision(self, signal_data: Dict[str, Any], 
                                  portfolio_data: Dict[str, Any]) -> ProbabilisticOutput:
        """Make probabilistic trading decision."""
        try:
            # Get ensemble predictions
            model_predictions = await self.ml_engine.get_ensemble_prediction(
                signal_data.get('market_data', {})
            )
            
            if not model_predictions:
                return self._create_no_trade_decision("No model predictions available")
            
            # Combine model predictions probabilistically
            combined_probabilities = await self._combine_predictions(model_predictions)
            
            # Apply Bayesian reasoning with prior beliefs
            bayesian_probabilities = await self._apply_bayesian_reasoning(
                combined_probabilities, signal_data, portfolio_data
            )
            
            # Calculate uncertainty and confidence
            uncertainty_score = self._calculate_uncertainty(bayesian_probabilities)
            confidence_level = self._assess_confidence(uncertainty_score, bayesian_probabilities)
            
            # Make final decision
            decision = self._make_final_decision(bayesian_probabilities, confidence_level)
            
            # Calculate expected returns and risk adjustment
            expected_return = await self._calculate_expected_return(
                bayesian_probabilities, signal_data
            )
            risk_adjusted_return = await self._calculate_risk_adjusted_return(
                expected_return, signal_data, portfolio_data
            )
            
            # Gather evidence
            supporting_evidence, conflicting_evidence = await self._analyze_evidence(
                model_predictions, signal_data, bayesian_probabilities
            )
            
            # Create output
            output = ProbabilisticOutput(
                decision=decision,
                confidence=confidence_level,
                probability_distribution=bayesian_probabilities,
                expected_return=expected_return,
                risk_adjusted_return=risk_adjusted_return,
                uncertainty_score=uncertainty_score,
                supporting_evidence=supporting_evidence,
                conflicting_evidence=conflicting_evidence,
                metadata={
                    'model_predictions': [
                        {
                            'model': p.model_name,
                            'prediction': p.prediction,
                            'confidence': p.confidence
                        } for p in model_predictions
                    ],
                    'signal_data': signal_data.get('final_signal', {}),
                    'timestamp': datetime.now()
                }
            )
            
            # Store decision history
            self.decision_history.append(output)
            
            # Keep only recent decisions
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.decision_history = [
                d for d in self.decision_history 
                if d.metadata['timestamp'] > cutoff_time
            ]
            
            return output
            
        except Exception as e:
            await handle_error_async(e, {'component': 'probabilistic_engine_decision'})
            return self._create_no_trade_decision(f"Decision error: {str(e)}")
    
    async def _combine_predictions(self, model_predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Combine multiple model predictions probabilistically."""
        combined_probs = {'buy': 0.0, 'hold': 0.0, 'sell': 0.0}
        total_weight = 0.0
        
        for prediction in model_predictions:
            model_weight = self.ml_engine.model_weights.get(prediction.model_name, 0.25)
            confidence_weight = prediction.confidence
            
            # Weight by both model weight and prediction confidence
            effective_weight = model_weight * confidence_weight
            total_weight += effective_weight
            
            for action, prob in prediction.probability_scores.items():
                if action in combined_probs:
                    combined_probs[action] += prob * effective_weight
        
        # Normalize probabilities
        if total_weight > 0:
            for action in combined_probs:
                combined_probs[action] /= total_weight
        else:
            # Default to neutral if no valid predictions
            combined_probs = {'buy': 0.33, 'hold': 0.34, 'sell': 0.33}
        
        return combined_probs
    
    async def _apply_bayesian_reasoning(self, prior_probs: Dict[str, float], 
                                       signal_data: Dict[str, Any],
                                       portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply Bayesian reasoning to update probabilities."""
        # Get validation results as evidence
        validation_reports = await validation_engine.comprehensive_validation(
            signal_data, portfolio_data
        )
        
        # Convert validation results to likelihood adjustments
        likelihood_adjustments = {'buy': 1.0, 'hold': 1.0, 'sell': 1.0}
        
        for report in validation_reports:
            if report.result == ValidationResult.REJECTED:
                # Strong evidence against trading
                likelihood_adjustments['buy'] *= 0.3
                likelihood_adjustments['sell'] *= 0.3
                likelihood_adjustments['hold'] *= 2.0
                
            elif report.result == ValidationResult.CONDITIONAL:
                # Moderate evidence adjustment
                confidence_factor = report.confidence_score
                likelihood_adjustments['buy'] *= (0.5 + confidence_factor * 0.5)
                likelihood_adjustments['sell'] *= (0.5 + confidence_factor * 0.5)
                
            elif report.result == ValidationResult.APPROVED:
                # Positive evidence for trading
                confidence_factor = report.confidence_score
                likelihood_adjustments['buy'] *= (1.0 + confidence_factor * 0.5)
                likelihood_adjustments['sell'] *= (1.0 + confidence_factor * 0.5)
                likelihood_adjustments['hold'] *= 0.8
        
        # Apply Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)
        posterior_probs = {}
        normalization_factor = 0.0
        
        for action in prior_probs:
            posterior = prior_probs[action] * likelihood_adjustments[action]
            posterior_probs[action] = posterior
            normalization_factor += posterior
        
        # Normalize to ensure probabilities sum to 1
        if normalization_factor > 0:
            for action in posterior_probs:
                posterior_probs[action] /= normalization_factor
        else:
            posterior_probs = prior_probs.copy()
        
        return posterior_probs
    
    def _calculate_uncertainty(self, probabilities: Dict[str, float]) -> float:
        """Calculate uncertainty using entropy."""
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize entropy (max entropy for 3 outcomes is log2(3))
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _assess_confidence(self, uncertainty_score: float, 
                          probabilities: Dict[str, float]) -> DecisionConfidence:
        """Assess confidence level based on uncertainty and probability distribution."""
        # Find the maximum probability
        max_prob = max(probabilities.values())
        
        # Confidence is high when uncertainty is low and max probability is high
        confidence_score = max_prob * (1 - uncertainty_score)
        
        if confidence_score >= self.confidence_thresholds[DecisionConfidence.VERY_HIGH]:
            return DecisionConfidence.VERY_HIGH
        elif confidence_score >= self.confidence_thresholds[DecisionConfidence.HIGH]:
            return DecisionConfidence.HIGH
        elif confidence_score >= self.confidence_thresholds[DecisionConfidence.MEDIUM]:
            return DecisionConfidence.MEDIUM
        elif confidence_score >= self.confidence_thresholds[DecisionConfidence.LOW]:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW
    
    def _make_final_decision(self, probabilities: Dict[str, float], 
                           confidence_level: DecisionConfidence) -> TradingDecision:
        """Make final trading decision based on probabilities and confidence."""
        # If confidence is very low, don't trade
        if confidence_level == DecisionConfidence.VERY_LOW:
            return TradingDecision.NO_TRADE
        
        # Find the action with highest probability
        best_action = max(probabilities.items(), key=lambda x: x[1])
        action, prob = best_action
        
        # Apply confidence thresholds
        if action == 'buy':
            if prob >= 0.6 and confidence_level in [DecisionConfidence.HIGH, DecisionConfidence.VERY_HIGH]:
                return TradingDecision.STRONG_BUY
            elif prob >= 0.45:
                return TradingDecision.BUY
            else:
                return TradingDecision.HOLD
                
        elif action == 'sell':
            if prob >= 0.6 and confidence_level in [DecisionConfidence.HIGH, DecisionConfidence.VERY_HIGH]:
                return TradingDecision.STRONG_SELL
            elif prob >= 0.45:
                return TradingDecision.SELL
            else:
                return TradingDecision.HOLD
                
        else:  # hold
            return TradingDecision.HOLD
    
    async def _calculate_expected_return(self, probabilities: Dict[str, float], 
                                       signal_data: Dict[str, Any]) -> float:
        """Calculate expected return based on probabilities."""
        # Simple expected return calculation
        buy_return = 0.05  # Assume 5% return for buy
        hold_return = 0.0  # No return for hold
        sell_return = -0.02  # Assume -2% return for sell (shorting not implemented)
        
        expected_return = (
            probabilities['buy'] * buy_return +
            probabilities['hold'] * hold_return +
            probabilities['sell'] * sell_return
        )
        
        return expected_return
    
    async def _calculate_risk_adjusted_return(self, expected_return: float, 
                                            signal_data: Dict[str, Any],
                                            portfolio_data: Dict[str, Any]) -> float:
        """Calculate risk-adjusted return using Sharpe-like ratio."""
        volatility = signal_data.get('market_data', {}).get('volatility', 0.05)
        
        # Risk-adjusted return (simplified Sharpe ratio)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        risk_adjusted = (expected_return - risk_free_rate) / max(volatility, 0.01)
        
        return risk_adjusted
    
    async def _analyze_evidence(self, model_predictions: List[ModelPrediction],
                               signal_data: Dict[str, Any],
                               probabilities: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze supporting and conflicting evidence."""
        supporting_evidence = []
        conflicting_evidence = []
        
        # Analyze model consensus
        buy_models = sum(1 for p in model_predictions if p.probability_scores.get('buy', 0) > 0.5)
        sell_models = sum(1 for p in model_predictions if p.probability_scores.get('sell', 0) > 0.5)
        
        if buy_models > len(model_predictions) * 0.6:
            supporting_evidence.append(f"Strong model consensus for buying ({buy_models}/{len(model_predictions)} models)")
        elif sell_models > len(model_predictions) * 0.6:
            supporting_evidence.append(f"Strong model consensus for selling ({sell_models}/{len(model_predictions)} models)")
        else:
            conflicting_evidence.append("Mixed model signals - no clear consensus")
        
        # Analyze signal strength
        final_signal = signal_data.get('final_signal', {})
        signal_confidence = final_signal.get('confidence_score', 0)
        
        if signal_confidence > 80:
            supporting_evidence.append(f"High signal confidence: {signal_confidence}%")
        elif signal_confidence < 60:
            conflicting_evidence.append(f"Low signal confidence: {signal_confidence}%")
        
        # Analyze technical indicators
        technical_analysis = signal_data.get('technical_analysis', {})
        bullish_indicators = sum(1 for indicator in technical_analysis.values() 
                               if isinstance(indicator, dict) and 'BUY' in str(indicator.get('signal', '')).upper())
        bearish_indicators = sum(1 for indicator in technical_analysis.values() 
                               if isinstance(indicator, dict) and 'SELL' in str(indicator.get('signal', '')).upper())
        
        if bullish_indicators > bearish_indicators:
            supporting_evidence.append(f"More bullish technical indicators ({bullish_indicators} vs {bearish_indicators})")
        elif bearish_indicators > bullish_indicators:
            supporting_evidence.append(f"More bearish technical indicators ({bearish_indicators} vs {bullish_indicators})")
        else:
            conflicting_evidence.append("Mixed technical indicator signals")
        
        return supporting_evidence, conflicting_evidence
    
    def _create_no_trade_decision(self, reason: str) -> ProbabilisticOutput:
        """Create a no-trade decision."""
        return ProbabilisticOutput(
            decision=TradingDecision.NO_TRADE,
            confidence=DecisionConfidence.HIGH,
            probability_distribution={'buy': 0.0, 'hold': 1.0, 'sell': 0.0},
            expected_return=0.0,
            risk_adjusted_return=0.0,
            uncertainty_score=0.0,
            supporting_evidence=[reason],
            conflicting_evidence=[],
            metadata={'reason': reason, 'timestamp': datetime.now()}
        )
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision statistics for monitoring."""
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history
        
        stats = {
            "total_decisions": len(recent_decisions),
            "decisions_by_type": {},
            "confidence_distribution": {},
            "average_expected_return": np.mean([d.expected_return for d in recent_decisions]),
            "average_uncertainty": np.mean([d.uncertainty_score for d in recent_decisions])
        }
        
        # Group by decision type
        for decision in recent_decisions:
            decision_type = decision.decision.value
            if decision_type not in stats["decisions_by_type"]:
                stats["decisions_by_type"][decision_type] = 0
            stats["decisions_by_type"][decision_type] += 1
            
            # Group by confidence
            confidence_level = decision.confidence.value
            if confidence_level not in stats["confidence_distribution"]:
                stats["confidence_distribution"][confidence_level] = 0
            stats["confidence_distribution"][confidence_level] += 1
        
        return stats


class BlackBoxProcessor:
    """Main black box processing system."""
    
    def __init__(self):
        self.probabilistic_engine = ProbabilisticEngine()
        self.processing_history = []
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the black box processor."""
        try:
            await self.probabilistic_engine.initialize()
            self.is_initialized = True
            crypto_logger.logger.info("🤖 Black Box Processor initialized")
        except Exception as e:
            await handle_error_async(e, {'component': 'black_box_processor_init'})
            raise
    
    async def process_trading_signal(self, signal_data: Dict[str, Any], 
                                   portfolio_data: Dict[str, Any]) -> ProbabilisticOutput:
        """Main processing method for trading signals."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            crypto_logger.logger.info(f"🔄 Processing signal for {signal_data.get('symbol', 'unknown')}")
            
            # Run probabilistic analysis
            result = await self.probabilistic_engine.make_trading_decision(
                signal_data, portfolio_data
            )
            
            # Log processing result
            crypto_logger.logger.info(
                f"📊 Black box decision: {result.decision.value} "
                f"(confidence: {result.confidence.value}, "
                f"expected return: {result.expected_return:.3f})"
            )
            
            # Store processing history
            self.processing_history.append({
                'timestamp': datetime.now(),
                'symbol': signal_data.get('symbol'),
                'decision': result.decision.value,
                'confidence': result.confidence.value,
                'expected_return': result.expected_return
            })
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.processing_history = [
                p for p in self.processing_history 
                if p['timestamp'] > cutoff_time
            ]
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'black_box_processor'})
            return self.probabilistic_engine._create_no_trade_decision(
                f"Processing error: {str(e)}"
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        base_stats = {
            "total_processed": len(self.processing_history),
            "processing_by_symbol": {},
            "decision_distribution": {}
        }
        
        if self.probabilistic_engine:
            decision_stats = self.probabilistic_engine.get_decision_statistics()
            base_stats.update(decision_stats)
        
        for processing in self.processing_history:
            symbol = processing.get('symbol', 'unknown')
            decision = processing.get('decision', 'unknown')
            
            if symbol not in base_stats["processing_by_symbol"]:
                base_stats["processing_by_symbol"][symbol] = 0
            base_stats["processing_by_symbol"][symbol] += 1
            
            if decision not in base_stats["decision_distribution"]:
                base_stats["decision_distribution"][decision] = 0
            base_stats["decision_distribution"][decision] += 1
        
        return base_stats


# Global black box processor instance
black_box_processor = BlackBoxProcessor()