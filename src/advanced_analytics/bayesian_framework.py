"""
Advanced Bayesian Probability Framework

Comprehensive Bayesian inference system for trading decisions with
exact implementation of probability theory and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async


@dataclass
class BayesianConfig:
    """Configuration for Bayesian inference framework."""
    
    # Performance constraints
    max_inference_time_ms: int = 100  # Target sub-500ms total
    max_posterior_samples: int = 10000
    enable_mcmc: bool = False  # Expensive, use for critical decisions only
    
    # Prior distributions
    default_prior_mean: float = 0.0
    default_prior_std: float = 1.0
    price_change_prior_std: float = 0.05  # 5% typical price change
    volatility_prior_alpha: float = 2.0
    volatility_prior_beta: float = 1.0
    
    # Likelihood parameters
    observation_noise_std: float = 0.01
    model_confidence_threshold: float = 0.7
    
    # Evidence integration
    enable_multi_source_evidence: bool = True
    evidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'technical_analysis': 0.3,
        'market_sentiment': 0.25,
        'volume_analysis': 0.2,
        'correlation_analysis': 0.15,
        'volatility_analysis': 0.1
    })
    
    # Uncertainty quantification
    confidence_intervals: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    enable_model_uncertainty: bool = True
    bayesian_model_averaging: bool = True


@dataclass
class PriorDistribution:
    """Prior distribution specification."""
    distribution_type: str  # 'normal', 'gamma', 'beta', 'uniform'
    parameters: Dict[str, float]
    description: str = ""


@dataclass
class Evidence:
    """Evidence for Bayesian updating."""
    source: str
    observation: Union[float, np.ndarray]
    likelihood_function: str  # 'normal', 'binomial', 'poisson'
    likelihood_params: Dict[str, float]
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PosteriorDistribution:
    """Posterior distribution result."""
    distribution_type: str
    parameters: Dict[str, float]
    samples: Optional[np.ndarray] = None
    confidence_intervals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    posterior_mean: float = 0.0
    posterior_std: float = 0.0
    log_evidence: float = 0.0  # Model evidence (marginal likelihood)


@dataclass
class BayesianInferenceResult:
    """Complete Bayesian inference result."""
    decision: str  # 'buy', 'sell', 'hold'
    posterior_probabilities: Dict[str, float]
    confidence_score: float
    uncertainty_score: float
    evidence_summary: Dict[str, Any]
    model_evidence: float
    decision_threshold: float
    expected_value: float
    value_at_risk: float
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class BayesianPriorEngine:
    """Engine for managing and updating prior distributions."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.priors = self._initialize_default_priors()
        self.prior_history = {}
    
    def _initialize_default_priors(self) -> Dict[str, PriorDistribution]:
        """Initialize default prior distributions for different parameters."""
        priors = {
            'price_direction': PriorDistribution(
                distribution_type='normal',
                parameters={'mu': 0.0, 'sigma': self.config.price_change_prior_std},
                description="Prior for price direction (log returns)"
            ),
            'volatility': PriorDistribution(
                distribution_type='gamma',
                parameters={'alpha': self.config.volatility_prior_alpha, 'beta': self.config.volatility_prior_beta},
                description="Prior for volatility (positive values)"
            ),
            'trend_strength': PriorDistribution(
                distribution_type='beta',
                parameters={'alpha': 2, 'beta': 2},
                description="Prior for trend strength (0-1)"
            ),
            'market_regime': PriorDistribution(
                distribution_type='uniform',
                parameters={'low': 0, 'high': 1},
                description="Prior for market regime classification"
            ),
            'signal_reliability': PriorDistribution(
                distribution_type='beta',
                parameters={'alpha': 5, 'beta': 2},
                description="Prior for signal reliability"
            )
        }
        
        return priors
    
    def get_prior(self, parameter_name: str) -> PriorDistribution:
        """Get prior distribution for a parameter."""
        return self.priors.get(parameter_name, 
                              PriorDistribution('normal', 
                                              {'mu': self.config.default_prior_mean, 
                                               'sigma': self.config.default_prior_std}))
    
    def update_prior_from_data(self, parameter_name: str, historical_data: np.ndarray):
        """Update prior distribution using historical data (empirical Bayes)."""
        try:
            if len(historical_data) < 10:  # Insufficient data
                return
            
            if parameter_name == 'price_direction':
                # Fit normal distribution to price changes
                mu_hat, sigma_hat = stats.norm.fit(historical_data)
                self.priors[parameter_name].parameters = {'mu': mu_hat, 'sigma': sigma_hat}
                
            elif parameter_name == 'volatility':
                # Fit gamma distribution to volatility data
                alpha_hat, _, beta_hat = stats.gamma.fit(historical_data, floc=0)
                self.priors[parameter_name].parameters = {'alpha': alpha_hat, 'beta': 1/beta_hat}
                
            elif parameter_name in ['trend_strength', 'signal_reliability']:
                # Fit beta distribution
                a_hat, b_hat, _, _ = stats.beta.fit(historical_data)
                self.priors[parameter_name].parameters = {'alpha': a_hat, 'beta': b_hat}
            
            crypto_logger.logger.debug(f"Updated prior for {parameter_name} from {len(historical_data)} data points")
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to update prior for {parameter_name}: {e}")
    
    def sample_from_prior(self, parameter_name: str, n_samples: int = 1000) -> np.ndarray:
        """Sample from prior distribution."""
        prior = self.get_prior(parameter_name)
        
        try:
            if prior.distribution_type == 'normal':
                return np.random.normal(prior.parameters['mu'], prior.parameters['sigma'], n_samples)
            elif prior.distribution_type == 'gamma':
                return np.random.gamma(prior.parameters['alpha'], 1/prior.parameters['beta'], n_samples)
            elif prior.distribution_type == 'beta':
                return np.random.beta(prior.parameters['alpha'], prior.parameters['beta'], n_samples)
            elif prior.distribution_type == 'uniform':
                return np.random.uniform(prior.parameters['low'], prior.parameters['high'], n_samples)
            else:
                return np.random.normal(0, 1, n_samples)
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to sample from prior {parameter_name}: {e}")
            return np.random.normal(0, 1, n_samples)


class BayesianLikelihoodEngine:
    """Engine for computing likelihood functions."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
    
    def compute_likelihood(self, evidence: Evidence, parameter_values: np.ndarray) -> np.ndarray:
        """Compute likelihood for given parameter values and evidence."""
        try:
            observation = evidence.observation
            params = evidence.likelihood_params
            
            if evidence.likelihood_function == 'normal':
                # Gaussian likelihood
                mu = parameter_values
                sigma = params.get('sigma', self.config.observation_noise_std)
                likelihood = stats.norm.pdf(observation, mu, sigma)
                
            elif evidence.likelihood_function == 'binomial':
                # For binary outcomes (success/failure signals)
                n = params.get('n', 1)
                p = np.clip(parameter_values, 1e-10, 1-1e-10)  # Avoid edge cases
                likelihood = stats.binom.pmf(observation, n, p)
                
            elif evidence.likelihood_function == 'gamma':
                # For positive continuous variables (volatility, etc.)
                alpha = params.get('alpha', 2)
                beta = parameter_values  # Parameter of interest
                beta = np.clip(beta, 1e-10, None)
                likelihood = stats.gamma.pdf(observation, alpha, scale=1/beta)
                
            elif evidence.likelihood_function == 'exponential':
                # For waiting times, durations
                rate = parameter_values
                rate = np.clip(rate, 1e-10, None)
                likelihood = stats.expon.pdf(observation, scale=1/rate)
                
            else:
                # Default to normal likelihood
                mu = parameter_values
                sigma = params.get('sigma', self.config.observation_noise_std)
                likelihood = stats.norm.pdf(observation, mu, sigma)
            
            # Apply evidence weight
            weighted_likelihood = likelihood ** evidence.weight
            
            return weighted_likelihood
            
        except Exception as e:
            crypto_logger.logger.warning(f"Likelihood computation failed for {evidence.source}: {e}")
            return np.ones_like(parameter_values) * 1e-10  # Small uniform likelihood


class BayesianInferenceEngine:
    """Main Bayesian inference engine with exact probability calculations."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.prior_engine = BayesianPriorEngine(config)
        self.likelihood_engine = BayesianLikelihoodEngine(config)
        
        self.inference_cache = {}
        self.model_evidence_history = []
    
    async def perform_inference(self, parameter_name: str, 
                               evidence_list: List[Evidence]) -> PosteriorDistribution:
        """Perform Bayesian inference to compute posterior distribution."""
        start_time = time.time()
        
        try:
            # Get prior distribution
            prior = self.prior_engine.get_prior(parameter_name)
            
            # Sample from prior
            n_samples = min(self.config.max_posterior_samples, 10000)
            prior_samples = self.prior_engine.sample_from_prior(parameter_name, n_samples)
            
            # Compute likelihood for each evidence
            log_likelihood_total = np.zeros(n_samples)
            
            for evidence in evidence_list:
                likelihood = self.likelihood_engine.compute_likelihood(evidence, prior_samples)
                # Use log-likelihood to avoid numerical underflow
                log_likelihood = np.log(np.clip(likelihood, 1e-300, None))
                log_likelihood_total += log_likelihood
            
            # Compute unnormalized log posterior
            log_posterior_unnorm = log_likelihood_total
            
            # Compute log evidence (marginal likelihood) for model comparison
            log_evidence = logsumexp(log_posterior_unnorm) - np.log(n_samples)
            
            # Normalize posterior (in log space for numerical stability)
            log_posterior = log_posterior_unnorm - logsumexp(log_posterior_unnorm)
            posterior_weights = np.exp(log_posterior)
            
            # Compute posterior statistics
            posterior_mean = np.average(prior_samples, weights=posterior_weights)
            posterior_var = np.average((prior_samples - posterior_mean)**2, weights=posterior_weights)
            posterior_std = np.sqrt(posterior_var)
            
            # Compute confidence intervals
            confidence_intervals = {}
            for confidence_level in self.config.confidence_intervals:
                alpha = 1 - confidence_level
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                # Weighted percentiles
                lower = self._weighted_percentile(prior_samples, posterior_weights, lower_percentile)
                upper = self._weighted_percentile(prior_samples, posterior_weights, upper_percentile)
                
                confidence_intervals[confidence_level] = (lower, upper)
            
            # Create posterior distribution
            posterior = PosteriorDistribution(
                distribution_type='empirical',  # Empirical distribution from samples
                parameters={
                    'mean': posterior_mean,
                    'std': posterior_std,
                    'log_evidence': log_evidence
                },
                samples=prior_samples,  # Store samples for further analysis
                confidence_intervals=confidence_intervals,
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                log_evidence=log_evidence
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Check performance constraint
            if processing_time > self.config.max_inference_time_ms:
                crypto_logger.logger.warning(
                    f"Bayesian inference took {processing_time:.2f}ms, "
                    f"exceeds target {self.config.max_inference_time_ms}ms"
                )
            
            crypto_logger.logger.debug(
                f"Bayesian inference for {parameter_name}: "
                f"mean={posterior_mean:.4f}, std={posterior_std:.4f}, "
                f"time={processing_time:.2f}ms"
            )
            
            return posterior
            
        except Exception as e:
            await handle_error_async(e, {'component': 'bayesian_inference', 'parameter': parameter_name})
            return self._create_default_posterior(parameter_name)
    
    def _weighted_percentile(self, data: np.ndarray, weights: np.ndarray, percentile: float) -> float:
        """Compute weighted percentile."""
        try:
            sorted_indices = np.argsort(data)
            sorted_data = data[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Compute cumulative weights
            cumsum_weights = np.cumsum(sorted_weights)
            cumsum_weights = cumsum_weights / cumsum_weights[-1]  # Normalize to [0, 1]
            
            # Find percentile
            percentile_fraction = percentile / 100.0
            index = np.searchsorted(cumsum_weights, percentile_fraction)
            
            if index == 0:
                return sorted_data[0]
            elif index >= len(sorted_data):
                return sorted_data[-1]
            else:
                # Linear interpolation
                frac = (percentile_fraction - cumsum_weights[index-1]) / (cumsum_weights[index] - cumsum_weights[index-1])
                return sorted_data[index-1] * (1 - frac) + sorted_data[index] * frac
                
        except Exception:
            return np.percentile(data, percentile)
    
    def _create_default_posterior(self, parameter_name: str) -> PosteriorDistribution:
        """Create default posterior for error cases."""
        return PosteriorDistribution(
            distribution_type='normal',
            parameters={'mean': 0.0, 'std': 1.0, 'log_evidence': -np.inf},
            confidence_intervals={},
            posterior_mean=0.0,
            posterior_std=1.0,
            log_evidence=-np.inf
        )


class BayesianDecisionEngine:
    """Decision-making engine using Bayesian inference results."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.decision_thresholds = {
            'buy': 0.6,   # 60% confidence for buy
            'sell': 0.6,  # 60% confidence for sell
            'strong_buy': 0.8,   # 80% confidence for strong buy
            'strong_sell': 0.8   # 80% confidence for strong sell
        }
    
    async def make_bayesian_decision(self, market_data: Dict[str, Any]) -> BayesianInferenceResult:
        """Make trading decision using comprehensive Bayesian analysis."""
        start_time = time.time()
        
        try:
            # Collect evidence from different sources
            evidence_list = await self._collect_evidence(market_data)
            
            if not evidence_list:
                return self._create_default_decision(start_time)
            
            # Perform Bayesian inference for key parameters
            inference_engine = BayesianInferenceEngine(self.config)
            
            # Infer price direction
            price_direction_evidence = [e for e in evidence_list if 'price' in e.source.lower()]
            price_posterior = await inference_engine.perform_inference('price_direction', price_direction_evidence)
            
            # Infer market volatility
            volatility_evidence = [e for e in evidence_list if 'volatility' in e.source.lower()]
            volatility_posterior = await inference_engine.perform_inference('volatility', volatility_evidence)
            
            # Infer signal reliability
            signal_evidence = [e for e in evidence_list if 'signal' in e.source.lower() or 'technical' in e.source.lower()]
            reliability_posterior = await inference_engine.perform_inference('signal_reliability', signal_evidence)
            
            # Compute decision probabilities
            decision_probs = self._compute_decision_probabilities(
                price_posterior, volatility_posterior, reliability_posterior
            )
            
            # Make final decision
            decision = self._make_final_decision(decision_probs)
            
            # Calculate confidence and uncertainty
            confidence_score = max(decision_probs.values())
            uncertainty_score = 1.0 - confidence_score
            
            # Calculate expected value and risk metrics
            expected_value = self._calculate_expected_value(price_posterior, decision_probs)
            value_at_risk = self._calculate_value_at_risk(price_posterior, volatility_posterior)
            
            # Summarize evidence
            evidence_summary = {
                'total_evidence_sources': len(evidence_list),
                'evidence_by_type': self._summarize_evidence_by_type(evidence_list),
                'average_evidence_weight': np.mean([e.weight for e in evidence_list]),
                'evidence_quality_score': self._assess_evidence_quality(evidence_list)
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            result = BayesianInferenceResult(
                decision=decision,
                posterior_probabilities=decision_probs,
                confidence_score=confidence_score,
                uncertainty_score=uncertainty_score,
                evidence_summary=evidence_summary,
                model_evidence=price_posterior.log_evidence + volatility_posterior.log_evidence,
                decision_threshold=self.decision_thresholds.get(decision, 0.5),
                expected_value=expected_value,
                value_at_risk=value_at_risk,
                processing_time_ms=processing_time
            )
            
            crypto_logger.logger.info(
                f"Bayesian decision: {decision} (confidence: {confidence_score:.3f}, "
                f"expected value: {expected_value:.4f})"
            )
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'bayesian_decision'})
            return self._create_default_decision(start_time)
    
    async def _collect_evidence(self, market_data: Dict[str, Any]) -> List[Evidence]:
        """Collect evidence from market data for Bayesian inference."""
        evidence_list = []
        
        try:
            # Technical analysis evidence
            if 'technical_analysis' in market_data:
                tech_data = market_data['technical_analysis']
                
                # RSI evidence
                if 'rsi' in tech_data:
                    rsi_value = tech_data['rsi'].get('value', 50)
                    # Convert RSI to price direction signal (-1 to 1)
                    rsi_signal = (rsi_value - 50) / 50
                    
                    evidence_list.append(Evidence(
                        source='technical_rsi',
                        observation=rsi_signal,
                        likelihood_function='normal',
                        likelihood_params={'sigma': 0.3},
                        weight=self.config.evidence_weights.get('technical_analysis', 0.3)
                    ))
                
                # MACD evidence
                if 'macd' in tech_data:
                    macd_signal = 1.0 if 'BUY' in tech_data['macd'].get('signal', '') else -1.0
                    
                    evidence_list.append(Evidence(
                        source='technical_macd',
                        observation=macd_signal,
                        likelihood_function='normal',
                        likelihood_params={'sigma': 0.4},
                        weight=self.config.evidence_weights.get('technical_analysis', 0.3)
                    ))
            
            # Volume analysis evidence
            if 'volume_analysis' in market_data:
                volume_data = market_data['volume_analysis']
                volume_signal = volume_data.get('volume_trend', 0)  # Normalized volume trend
                
                evidence_list.append(Evidence(
                    source='volume_trend',
                    observation=volume_signal,
                    likelihood_function='normal',
                    likelihood_params={'sigma': 0.5},
                    weight=self.config.evidence_weights.get('volume_analysis', 0.2)
                ))
            
            # Price momentum evidence
            if 'final_signal' in market_data:
                signal_data = market_data['final_signal']
                confidence = signal_data.get('confidence_score', 50) / 100
                signal_direction = 1.0 if signal_data.get('signal') in ['BUY', 'STRONG_BUY'] else -1.0
                
                evidence_list.append(Evidence(
                    source='signal_confidence',
                    observation=confidence,
                    likelihood_function='beta',
                    likelihood_params={'alpha': 2, 'beta': 2},
                    weight=confidence  # Weight by signal confidence
                ))
                
                evidence_list.append(Evidence(
                    source='price_signal_direction',
                    observation=signal_direction,
                    likelihood_function='normal',
                    likelihood_params={'sigma': 0.3},
                    weight=confidence
                ))
            
            # Market volatility evidence
            if 'market_data' in market_data:
                market = market_data['market_data']
                volatility = market.get('volatility', 0.05)
                
                evidence_list.append(Evidence(
                    source='volatility_observation',
                    observation=volatility,
                    likelihood_function='gamma',
                    likelihood_params={'alpha': 2},
                    weight=self.config.evidence_weights.get('volatility_analysis', 0.1)
                ))
            
            return evidence_list
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to collect evidence: {e}")
            return []
    
    def _compute_decision_probabilities(self, price_posterior: PosteriorDistribution,
                                      volatility_posterior: PosteriorDistribution,
                                      reliability_posterior: PosteriorDistribution) -> Dict[str, float]:
        """Compute probabilities for each decision based on posterior distributions."""
        try:
            # Sample from posteriors
            n_samples = 1000
            
            price_samples = price_posterior.samples[:n_samples] if price_posterior.samples is not None else np.random.normal(price_posterior.posterior_mean, price_posterior.posterior_std, n_samples)
            volatility_samples = volatility_posterior.samples[:n_samples] if volatility_posterior.samples is not None else np.random.gamma(2, 1/2, n_samples)
            reliability_samples = reliability_posterior.samples[:n_samples] if reliability_posterior.samples is not None else np.random.beta(2, 2, n_samples)
            
            # Compute decision scores for each sample
            buy_scores = []
            sell_scores = []
            hold_scores = []
            
            for price, vol, rel in zip(price_samples, volatility_samples, reliability_samples):
                # Adjust for reliability
                adjusted_price = price * rel
                
                # Adjust for volatility (higher volatility reduces confidence)
                volatility_penalty = min(1.0, vol / 0.1)  # Normalize volatility
                confidence_adjustment = 1.0 / (1.0 + volatility_penalty)
                
                final_score = adjusted_price * confidence_adjustment
                
                # Convert to decision probabilities
                if final_score > 0.1:
                    buy_scores.append(final_score)
                    sell_scores.append(0)
                    hold_scores.append(1 - abs(final_score))
                elif final_score < -0.1:
                    buy_scores.append(0)
                    sell_scores.append(abs(final_score))
                    hold_scores.append(1 - abs(final_score))
                else:
                    buy_scores.append(0)
                    sell_scores.append(0)
                    hold_scores.append(1)
            
            # Compute average probabilities
            buy_prob = np.mean(buy_scores)
            sell_prob = np.mean(sell_scores)
            hold_prob = np.mean(hold_scores)
            
            # Normalize probabilities
            total = buy_prob + sell_prob + hold_prob
            if total > 0:
                buy_prob /= total
                sell_prob /= total
                hold_prob /= total
            else:
                buy_prob = sell_prob = hold_prob = 1/3
            
            return {
                'buy': float(buy_prob),
                'sell': float(sell_prob),
                'hold': float(hold_prob)
            }
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to compute decision probabilities: {e}")
            return {'buy': 1/3, 'sell': 1/3, 'hold': 1/3}
    
    def _make_final_decision(self, probabilities: Dict[str, float]) -> str:
        """Make final decision based on probabilities and thresholds."""
        max_prob = max(probabilities.values())
        
        # Find decision with maximum probability
        decision = max(probabilities, key=probabilities.get)
        
        # Check if probability exceeds threshold
        if decision in ['buy', 'sell']:
            if max_prob >= self.decision_thresholds.get('strong_' + decision, 0.8):
                return 'strong_' + decision
            elif max_prob >= self.decision_thresholds.get(decision, 0.6):
                return decision
            else:
                return 'hold'  # Not confident enough
        
        return decision
    
    def _calculate_expected_value(self, price_posterior: PosteriorDistribution, 
                                decision_probs: Dict[str, float]) -> float:
        """Calculate expected value of the trading decision."""
        try:
            # Simplified expected value calculation
            expected_return = price_posterior.posterior_mean
            
            # Weight by decision probabilities
            buy_weight = decision_probs.get('buy', 0)
            sell_weight = decision_probs.get('sell', 0)
            
            # Expected value (simplified)
            expected_value = buy_weight * expected_return - sell_weight * expected_return
            
            return float(expected_value)
            
        except Exception:
            return 0.0
    
    def _calculate_value_at_risk(self, price_posterior: PosteriorDistribution,
                               volatility_posterior: PosteriorDistribution) -> float:
        """Calculate Value at Risk (95% confidence level)."""
        try:
            # Use 95% confidence interval for VaR
            if 0.95 in price_posterior.confidence_intervals:
                lower_bound = price_posterior.confidence_intervals[0.95][0]
                return abs(lower_bound)
            else:
                # Approximate VaR using normal approximation
                return abs(price_posterior.posterior_mean - 1.96 * price_posterior.posterior_std)
                
        except Exception:
            return 0.05  # Default 5% VaR
    
    def _summarize_evidence_by_type(self, evidence_list: List[Evidence]) -> Dict[str, int]:
        """Summarize evidence by type."""
        evidence_types = {}
        for evidence in evidence_list:
            evidence_type = evidence.source.split('_')[0]  # Get first part of source name
            evidence_types[evidence_type] = evidence_types.get(evidence_type, 0) + 1
        return evidence_types
    
    def _assess_evidence_quality(self, evidence_list: List[Evidence]) -> float:
        """Assess overall quality of evidence (0-1 score)."""
        if not evidence_list:
            return 0.0
        
        # Quality based on number of sources, weights, and recency
        source_diversity = len(set(e.source for e in evidence_list))
        avg_weight = np.mean([e.weight for e in evidence_list])
        
        # Recency score (evidence from last hour is best)
        now = datetime.now()
        recency_scores = []
        for evidence in evidence_list:
            hours_old = (now - evidence.timestamp).seconds / 3600
            recency_score = max(0, 1 - hours_old)  # Linear decay
            recency_scores.append(recency_score)
        
        avg_recency = np.mean(recency_scores)
        
        # Combined quality score
        quality_score = (source_diversity / 10 + avg_weight + avg_recency) / 3
        
        return min(1.0, quality_score)
    
    def _create_default_decision(self, start_time: float) -> BayesianInferenceResult:
        """Create default decision for error cases."""
        return BayesianInferenceResult(
            decision='hold',
            posterior_probabilities={'buy': 1/3, 'sell': 1/3, 'hold': 1/3},
            confidence_score=0.33,
            uncertainty_score=0.67,
            evidence_summary={'total_evidence_sources': 0},
            model_evidence=-np.inf,
            decision_threshold=0.5,
            expected_value=0.0,
            value_at_risk=0.05,
            processing_time_ms=(time.time() - start_time) * 1000
        )


class AdvancedBayesianFramework:
    """Main Bayesian framework integrating all components."""
    
    def __init__(self, config: Optional[BayesianConfig] = None):
        self.config = config or BayesianConfig()
        self.decision_engine = BayesianDecisionEngine(self.config)
        self.inference_engine = BayesianInferenceEngine(self.config)
        
        self.decision_history = []
        
        crypto_logger.logger.info("🧠 Advanced Bayesian Framework initialized")
    
    async def analyze_market_bayesian(self, market_data: Dict[str, Any]) -> BayesianInferenceResult:
        """Perform comprehensive Bayesian analysis of market data."""
        try:
            result = await self.decision_engine.make_bayesian_decision(market_data)
            
            # Store decision history
            self.decision_history.append(result)
            
            # Limit history size
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'bayesian_framework'})
            return self.decision_engine._create_default_decision(time.time())
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get Bayesian framework performance statistics."""
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        processing_times = [d.processing_time_ms for d in recent_decisions]
        confidence_scores = [d.confidence_score for d in recent_decisions]
        
        decision_distribution = {}
        for decision in recent_decisions:
            dec = decision.decision
            decision_distribution[dec] = decision_distribution.get(dec, 0) + 1
        
        return {
            'total_decisions': len(self.decision_history),
            'avg_processing_time_ms': np.mean(processing_times),
            'avg_confidence_score': np.mean(confidence_scores),
            'decision_distribution': decision_distribution,
            'performance_target_met': np.mean(processing_times) < self.config.max_inference_time_ms,
            'high_confidence_decisions': sum(1 for d in recent_decisions if d.confidence_score > 0.7) / len(recent_decisions)
        }
    
    def update_config(self, new_config: BayesianConfig):
        """Update framework configuration."""
        self.config = new_config
        self.decision_engine = BayesianDecisionEngine(new_config)
        self.inference_engine = BayesianInferenceEngine(new_config)
        crypto_logger.logger.info("Bayesian framework configuration updated")


# Global Bayesian framework instance
bayesian_framework = AdvancedBayesianFramework()