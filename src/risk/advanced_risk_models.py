import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class RiskMetrics:
    portfolio_value: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    maximum_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_to_market: float
    risk_adjusted_return: float
    tail_risk: float
    fat_tail_measure: float
    stress_test_results: Dict[str, float]
    monte_carlo_results: Dict[str, Any]
    timestamp: datetime

@dataclass
class PortfolioPosition:
    symbol: str
    quantity: float
    market_value: float
    weight: float
    beta: float
    volatility: float
    correlation_matrix: Dict[str, float]
    var_contribution: float
    marginal_var: float
    component_var: float

class AdvancedRiskModels:
    """Advanced risk modeling and measurement system."""
    
    def __init__(self):
        self.config = Config()
        self.historical_data = {}
        self.correlation_matrix = pd.DataFrame()
        self.covariance_matrix = pd.DataFrame()
        self.risk_free_rate = 0.05  # 5% annual risk-free rate
        self.market_benchmark = 'BTC'  # Use BTC as market benchmark
        
        # Risk model parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_periods = {
            'short_term': 30,   # 30 days
            'medium_term': 90,  # 90 days
            'long_term': 252    # 1 year
        }
        
        # Monte Carlo parameters
        self.mc_simulations = 10000
        self.mc_time_horizon = 252  # 1 year
        
    async def initialize_risk_models(self):
        """Initialize risk modeling system."""
        crypto_logger.logger.info("Initializing advanced risk models")
        
        try:
            # Generate mock historical data for testing
            await self._generate_mock_historical_data()
            
            # Calculate correlation and covariance matrices
            await self._calculate_correlation_matrices()
            
            # Initialize risk factor models
            await self._initialize_risk_factors()
            
            crypto_logger.logger.info("✓ Advanced risk models initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing risk models: {e}")
    
    async def _generate_mock_historical_data(self):
        """Generate mock historical price data for risk calculations."""
        
        # Crypto assets to model
        assets = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'AVAX']
        
        # Generate correlated returns
        np.random.seed(42)  # For reproducible results
        n_days = 500
        n_assets = len(assets)
        
        # Create correlation matrix (assets more correlated during market stress)
        base_correlation = 0.3
        correlation_matrix = np.full((n_assets, n_assets), base_correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Add some specific correlations
        correlation_matrix[0, 1] = 0.7  # BTC-ETH high correlation
        correlation_matrix[1, 0] = 0.7
        
        # Generate returns with fat tails (t-distribution)
        degrees_freedom = 5  # Heavy tails
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=correlation_matrix,
            size=n_days
        )
        
        # Add fat tails by mixing with t-distribution
        t_returns = np.random.standard_t(df=degrees_freedom, size=(n_days, n_assets))
        returns = 0.8 * returns + 0.2 * t_returns * 0.1  # Mix normal and t-distribution
        
        # Scale returns to realistic levels
        annual_vols = [0.8, 0.9, 1.2, 1.1, 1.0, 0.7, 0.9, 1.0]  # Annual volatilities
        daily_vols = [vol / np.sqrt(252) for vol in annual_vols]
        
        for i in range(n_assets):
            returns[:, i] = returns[:, i] * daily_vols[i]
        
        # Add market regime changes (volatility clustering)
        regime_changes = np.random.poisson(0.05, n_days)  # Low probability of regime change
        volatility_multiplier = np.ones(n_days)
        
        for i in range(1, n_days):
            if regime_changes[i] > 0:
                volatility_multiplier[i:] *= np.random.uniform(0.5, 2.0)  # Regime change
        
        # Apply volatility clustering
        for i in range(n_assets):
            returns[:, i] = returns[:, i] * volatility_multiplier
        
        # Create price series
        initial_prices = [45000, 2500, 1.0, 100, 0.8, 25, 30, 80]  # Starting prices
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        for i, asset in enumerate(assets):
            prices = [initial_prices[i]]
            for ret in returns[:, i]:
                prices.append(prices[-1] * (1 + ret))
            
            self.historical_data[asset] = pd.DataFrame({
                'date': dates,
                'price': prices[1:],  # Skip the initial price
                'returns': returns[:, i]
            }).set_index('date')
        
        crypto_logger.logger.info(f"Generated {n_days} days of mock data for {len(assets)} assets")
    
    async def _calculate_correlation_matrices(self):
        """Calculate correlation and covariance matrices."""
        
        if not self.historical_data:
            return
        
        # Create returns matrix
        returns_data = {}
        for asset, data in self.historical_data.items():
            returns_data[asset] = data['returns']
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        # Calculate covariance matrix with shrinkage estimation
        shrinkage_estimator = LedoitWolf()
        shrinkage_cov = shrinkage_estimator.fit(returns_df.dropna()).covariance_
        
        self.covariance_matrix = pd.DataFrame(
            shrinkage_cov, 
            index=returns_df.columns, 
            columns=returns_df.columns
        )
        
        crypto_logger.logger.info(f"Calculated correlation matrix for {len(returns_df.columns)} assets")
    
    async def _initialize_risk_factors(self):
        """Initialize risk factor models (PCA, etc.)."""
        
        if self.historical_data:
            returns_data = pd.DataFrame({
                asset: data['returns'] for asset, data in self.historical_data.items()
            }).dropna()
            
            # Principal Component Analysis for risk factors
            pca = PCA(n_components=min(5, len(returns_data.columns)))
            pca.fit(returns_data)
            
            self.risk_factors = {
                'pca_model': pca,
                'explained_variance': pca.explained_variance_ratio_,
                'n_components': pca.n_components_
            }
            
            crypto_logger.logger.info(f"Initialized PCA risk factors explaining {pca.explained_variance_ratio_.sum():.2%} of variance")
    
    async def calculate_portfolio_var(self, positions: List[Dict[str, Any]], confidence_level: float = 0.95, 
                                   holding_period: int = 1) -> Dict[str, Any]:
        """Calculate Value at Risk using multiple methodologies."""
        
        # Validate positions
        if not positions:
            return {'error': 'No positions provided'}
        
        portfolio_value = sum(pos['market_value'] for pos in positions)
        
        # Get portfolio returns
        portfolio_returns = await self._calculate_portfolio_returns(positions)
        
        if portfolio_returns is None or len(portfolio_returns) < 30:
            return {'error': 'Insufficient historical data for VaR calculation'}
        
        # Calculate VaR using different methods
        var_results = {}
        
        # 1. Historical Simulation VaR
        var_results['historical'] = self._calculate_historical_var(
            portfolio_returns, confidence_level, holding_period
        )
        
        # 2. Parametric VaR (assumes normal distribution)
        var_results['parametric'] = self._calculate_parametric_var(
            portfolio_returns, confidence_level, holding_period
        )
        
        # 3. Monte Carlo VaR
        var_results['monte_carlo'] = await self._calculate_monte_carlo_var(
            positions, confidence_level, holding_period
        )
        
        # 4. Filtered Historical Simulation (EWMA)
        var_results['ewma'] = self._calculate_ewma_var(
            portfolio_returns, confidence_level, holding_period
        )
        
        # Calculate Expected Shortfall (CVaR)
        cvar = self._calculate_expected_shortfall(portfolio_returns, confidence_level)
        
        return {
            'portfolio_value': portfolio_value,
            'confidence_level': confidence_level,
            'holding_period_days': holding_period,
            'var_estimates': var_results,
            'expected_shortfall': cvar,
            'var_summary': {
                'conservative_var': max(var_results.values()),
                'average_var': np.mean(list(var_results.values())),
                'model_dispersion': np.std(list(var_results.values()))
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float, 
                                holding_period: int) -> float:
        """Calculate VaR using historical simulation."""
        
        # Scale returns for holding period
        scaled_returns = returns * np.sqrt(holding_period)
        
        # Calculate percentile
        alpha = 1 - confidence_level
        var = -np.percentile(scaled_returns, alpha * 100)
        
        return var
    
    def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float, 
                                holding_period: int) -> float:
        """Calculate VaR assuming normal distribution."""
        
        # Calculate portfolio statistics
        mean_return = np.mean(returns)
        portfolio_vol = np.std(returns)
        
        # Scale for holding period
        scaled_mean = mean_return * holding_period
        scaled_vol = portfolio_vol * np.sqrt(holding_period)
        
        # Calculate VaR
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var = -(scaled_mean + z_score * scaled_vol)
        
        return var
    
    async def _calculate_monte_carlo_var(self, positions: List[Dict[str, Any]], 
                                       confidence_level: float, holding_period: int) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        
        try:
            # Get asset weights
            portfolio_value = sum(pos['market_value'] for pos in positions)
            weights = np.array([pos['market_value'] / portfolio_value for pos in positions])
            assets = [pos['symbol'] for pos in positions]
            
            # Get historical statistics
            returns_data = []
            for asset in assets:
                if asset in self.historical_data:
                    returns_data.append(self.historical_data[asset]['returns'].values)
                else:
                    # Use market return if asset not found
                    returns_data.append(self.historical_data['BTC']['returns'].values)
            
            returns_matrix = np.column_stack(returns_data)
            mean_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)
            
            # Monte Carlo simulation
            np.random.seed(None)  # Reset seed for true randomness
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, size=self.mc_simulations
            )
            
            # Calculate portfolio returns for each simulation
            portfolio_returns = np.dot(simulated_returns, weights)
            
            # Scale for holding period
            scaled_returns = portfolio_returns * np.sqrt(holding_period)
            
            # Calculate VaR
            alpha = 1 - confidence_level
            var = -np.percentile(scaled_returns, alpha * 100)
            
            return var
            
        except Exception as e:
            crypto_logger.logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return 0.0
    
    def _calculate_ewma_var(self, returns: np.ndarray, confidence_level: float, 
                          holding_period: int, lambda_param: float = 0.94) -> float:
        """Calculate VaR using Exponentially Weighted Moving Average."""
        
        # Calculate EWMA variance
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        # Calculate weighted mean and variance
        weighted_mean = np.sum(weights * returns)
        weighted_var = np.sum(weights * (returns - weighted_mean) ** 2)
        
        # Scale for holding period
        scaled_mean = weighted_mean * holding_period
        scaled_vol = np.sqrt(weighted_var * holding_period)
        
        # Calculate VaR
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var = -(scaled_mean + z_score * scaled_vol)
        
        return var
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        
        alpha = 1 - confidence_level
        var_threshold = -np.percentile(returns, alpha * 100)
        
        # Calculate average of returns beyond VaR
        tail_returns = returns[returns <= -var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        expected_shortfall = -np.mean(tail_returns)
        return expected_shortfall
    
    async def _calculate_portfolio_returns(self, positions: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Calculate historical portfolio returns from positions."""
        
        try:
            portfolio_value = sum(pos['market_value'] for pos in positions)
            weights = {pos['symbol']: pos['market_value'] / portfolio_value for pos in positions}
            
            # Get aligned return data
            return_series = []
            min_length = float('inf')
            
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.historical_data:
                    returns = self.historical_data[symbol]['returns']
                    return_series.append(returns)
                    min_length = min(min_length, len(returns))
                else:
                    # Use BTC as proxy if symbol not found
                    returns = self.historical_data.get('BTC', pd.Series()).get('returns', pd.Series())
                    return_series.append(returns)
                    min_length = min(min_length, len(returns))
            
            if min_length == float('inf') or min_length == 0:
                return None
            
            # Align all series to same length
            aligned_returns = []
            for returns in return_series:
                if len(returns) > min_length:
                    aligned_returns.append(returns.iloc[-min_length:].values)
                else:
                    aligned_returns.append(returns.values)
            
            # Calculate portfolio returns
            returns_matrix = np.column_stack(aligned_returns)
            weight_array = np.array([weights.get(pos['symbol'], 0) for pos in positions])
            portfolio_returns = np.dot(returns_matrix, weight_array)
            
            return portfolio_returns
            
        except Exception as e:
            crypto_logger.logger.error(f"Error calculating portfolio returns: {e}")
            return None
    
    async def run_stress_tests(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive stress tests on portfolio."""
        
        portfolio_value = sum(pos['market_value'] for pos in positions)
        stress_scenarios = {}
        
        # 1. Market crash scenarios
        crash_scenarios = {
            'mild_crash': -0.15,      # -15% market drop
            'severe_crash': -0.30,    # -30% market drop
            'black_swan': -0.50       # -50% market drop
        }
        
        for scenario_name, market_drop in crash_scenarios.items():
            scenario_pnl = 0
            for pos in positions:
                # Assume crypto assets move with market (beta = 1.0 for simplicity)
                asset_drop = market_drop * pos.get('beta', 1.0)
                pos_pnl = pos['market_value'] * asset_drop
                scenario_pnl += pos_pnl
            
            stress_scenarios[scenario_name] = {
                'market_drop': market_drop,
                'portfolio_pnl': scenario_pnl,
                'portfolio_pnl_pct': scenario_pnl / portfolio_value,
                'new_portfolio_value': portfolio_value + scenario_pnl
            }
        
        # 2. Volatility shock scenarios
        vol_scenarios = {
            'vol_spike_50': 1.5,      # 50% increase in volatility
            'vol_spike_100': 2.0,     # 100% increase in volatility
            'vol_spike_200': 3.0      # 200% increase in volatility
        }
        
        for scenario_name, vol_multiplier in vol_scenarios.items():
            # Estimate impact on options/derivatives positions
            vol_impact = 0
            for pos in positions:
                # Simple approximation: higher vol increases option values
                if 'option' in pos.get('type', '').lower():
                    vega = pos.get('vega', 0)  # Vega sensitivity
                    vol_change = (vol_multiplier - 1) * 100  # Percentage points
                    vol_impact += vega * vol_change / 100
            
            stress_scenarios[f'volatility_{scenario_name}'] = {
                'vol_multiplier': vol_multiplier,
                'vol_impact_pnl': vol_impact,
                'description': f'Volatility increases by {(vol_multiplier-1)*100:.0f}%'
            }
        
        # 3. Correlation breakdown scenario
        correlation_breakdown = {
            'correlation_breakdown': {
                'scenario': 'All asset correlations spike to 0.9 during crisis',
                'impact': 'Diversification benefits disappear',
                'estimated_var_increase': '25-50%'
            }
        }
        
        # 4. Liquidity crisis scenario
        liquidity_crisis = {
            'liquidity_crisis': {
                'scenario': 'Bid-ask spreads widen by 300-500%',
                'trading_cost_increase': portfolio_value * 0.02,  # 2% of portfolio value
                'liquidation_discount': portfolio_value * 0.05    # 5% fire sale discount
            }
        }
        
        return {
            'portfolio_value': portfolio_value,
            'stress_scenarios': {
                **stress_scenarios,
                **correlation_breakdown,
                **liquidity_crisis
            },
            'worst_case_scenario': min(
                [s['portfolio_pnl'] for s in stress_scenarios.values() if isinstance(s, dict) and 'portfolio_pnl' in s],
                default=0
            ),
            'stress_test_summary': {
                'scenarios_tested': len(stress_scenarios) + 2,
                'max_loss_scenario': min(stress_scenarios.keys(), key=lambda k: stress_scenarios[k].get('portfolio_pnl', 0)),
                'max_loss_amount': min([s.get('portfolio_pnl', 0) for s in stress_scenarios.values()]),
                'max_loss_percentage': min([s.get('portfolio_pnl_pct', 0) for s in stress_scenarios.values()]) * 100
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def monte_carlo_simulation(self, positions: List[Dict[str, Any]], 
                                   time_horizon_days: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio performance."""
        
        try:
            portfolio_value = sum(pos['market_value'] for pos in positions)
            assets = [pos['symbol'] for pos in positions]
            weights = np.array([pos['market_value'] / portfolio_value for pos in positions])
            
            # Get historical data
            returns_data = []
            for asset in assets:
                if asset in self.historical_data:
                    returns_data.append(self.historical_data[asset]['returns'].values)
                else:
                    returns_data.append(self.historical_data['BTC']['returns'].values)
            
            returns_matrix = np.column_stack(returns_data)
            mean_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)
            
            # Monte Carlo simulation
            np.random.seed(None)
            simulation_results = []
            
            for _ in range(self.mc_simulations):
                # Generate random returns for time horizon
                random_returns = np.random.multivariate_normal(
                    mean_returns, cov_matrix, size=time_horizon_days
                )
                
                # Calculate portfolio returns
                portfolio_returns = np.dot(random_returns, weights)
                
                # Calculate cumulative return
                cumulative_return = np.prod(1 + portfolio_returns) - 1
                final_value = portfolio_value * (1 + cumulative_return)
                
                simulation_results.append({
                    'final_value': final_value,
                    'total_return': cumulative_return,
                    'max_drawdown': self._calculate_drawdown(portfolio_returns)
                })
            
            # Analyze results
            final_values = [result['final_value'] for result in simulation_results]
            total_returns = [result['total_return'] for result in simulation_results]
            max_drawdowns = [result['max_drawdown'] for result in simulation_results]
            
            return {
                'simulation_parameters': {
                    'initial_portfolio_value': portfolio_value,
                    'time_horizon_days': time_horizon_days,
                    'number_of_simulations': self.mc_simulations
                },
                'results_summary': {
                    'expected_final_value': np.mean(final_values),
                    'expected_return': np.mean(total_returns),
                    'return_volatility': np.std(total_returns),
                    'median_final_value': np.median(final_values),
                    'value_at_risk_5%': np.percentile(final_values, 5),
                    'value_at_risk_1%': np.percentile(final_values, 1),
                    'probability_of_loss': sum(1 for r in total_returns if r < 0) / len(total_returns),
                    'probability_of_50%_gain': sum(1 for r in total_returns if r > 0.5) / len(total_returns),
                    'average_max_drawdown': np.mean(max_drawdowns),
                    'worst_max_drawdown': max(max_drawdowns)
                },
                'distribution_percentiles': {
                    f'{p}%': np.percentile(final_values, p) 
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                },
                'risk_metrics': {
                    'sharpe_ratio': np.mean(total_returns) / np.std(total_returns) * np.sqrt(252) if np.std(total_returns) > 0 else 0,
                    'sortino_ratio': self._calculate_sortino_ratio(total_returns),
                    'maximum_drawdown_95%': np.percentile(max_drawdowns, 95),
                    'tail_ratio': np.percentile(total_returns, 95) / abs(np.percentile(total_returns, 5))
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            crypto_logger.logger.error(f"Monte Carlo simulation failed: {e}")
            return {'error': f'Simulation failed: {e}'}
    
    def _calculate_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from return series."""
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate Sortino ratio (downside deviation adjusted)."""
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return float('inf')
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    async def calculate_risk_adjusted_performance(self, positions: List[Dict[str, Any]], 
                                                benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk-adjusted performance metrics."""
        
        # Get portfolio returns
        portfolio_returns = await self._calculate_portfolio_returns(positions)
        
        if portfolio_returns is None:
            return {'error': 'Unable to calculate portfolio returns'}
        
        # Use BTC as benchmark if not provided
        if benchmark_returns is None and 'BTC' in self.historical_data:
            benchmark_returns = self.historical_data['BTC']['returns'].values[-len(portfolio_returns):]
        
        # Calculate performance metrics
        portfolio_value = sum(pos['market_value'] for pos in positions)
        
        # Basic statistics
        mean_return = np.mean(portfolio_returns) * 252  # Annualized
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        # Risk-adjusted returns
        sharpe_ratio = (mean_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        
        # Beta and correlation to benchmark
        if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            correlation = 0
            beta = 1
        
        # Maximum drawdown
        max_drawdown = self._calculate_drawdown(portfolio_returns)
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR calculations
        var_95 = self._calculate_historical_var(portfolio_returns, 0.95, 1)
        var_99 = self._calculate_historical_var(portfolio_returns, 0.99, 1)
        cvar_95 = self._calculate_expected_shortfall(portfolio_returns, 0.95)
        cvar_99 = self._calculate_expected_shortfall(portfolio_returns, 0.99)
        
        # Tail risk measures
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        return {
            'portfolio_value': portfolio_value,
            'performance_metrics': {
                'annualized_return': mean_return,
                'annualized_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'beta': beta,
                'correlation_to_benchmark': correlation,
                'maximum_drawdown': max_drawdown
            },
            'risk_metrics': {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_ratio': abs(np.percentile(portfolio_returns, 95) / np.percentile(portfolio_returns, 5)) if np.percentile(portfolio_returns, 5) != 0 else 0
            },
            'risk_assessment': {
                'overall_risk_level': self._assess_risk_level(var_95, max_drawdown, portfolio_vol),
                'fat_tail_risk': 'High' if kurtosis > 3 else 'Medium' if kurtosis > 1 else 'Low',
                'downside_skew': 'Negative' if skewness < -0.5 else 'Neutral' if abs(skewness) <= 0.5 else 'Positive'
            },
            'recommendations': self._generate_risk_recommendations(sharpe_ratio, max_drawdown, var_95),
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_risk_level(self, var_95: float, max_drawdown: float, volatility: float) -> str:
        """Assess overall portfolio risk level."""
        
        risk_score = 0
        
        # VaR contribution
        if var_95 > 0.05:  # > 5% daily VaR
            risk_score += 3
        elif var_95 > 0.03:  # > 3% daily VaR
            risk_score += 2
        elif var_95 > 0.02:  # > 2% daily VaR
            risk_score += 1
        
        # Drawdown contribution
        if max_drawdown > 0.3:  # > 30% drawdown
            risk_score += 3
        elif max_drawdown > 0.2:  # > 20% drawdown
            risk_score += 2
        elif max_drawdown > 0.1:  # > 10% drawdown
            risk_score += 1
        
        # Volatility contribution
        if volatility > 0.6:  # > 60% annual vol
            risk_score += 3
        elif volatility > 0.4:  # > 40% annual vol
            risk_score += 2
        elif volatility > 0.25:  # > 25% annual vol
            risk_score += 1
        
        if risk_score >= 7:
            return 'Very High'
        elif risk_score >= 5:
            return 'High'
        elif risk_score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_risk_recommendations(self, sharpe_ratio: float, max_drawdown: float, var_95: float) -> List[str]:
        """Generate risk management recommendations."""
        
        recommendations = []
        
        if sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk-adjusted returns through better asset selection or position sizing")
        
        if max_drawdown > 0.2:
            recommendations.append("Implement stricter stop-loss rules to limit maximum drawdown")
        
        if var_95 > 0.05:
            recommendations.append("Daily VaR exceeds 5% - consider reducing position sizes or adding hedges")
        
        if sharpe_ratio < 1.0:
            recommendations.append("Sharpe ratio below 1.0 indicates sub-optimal risk-adjusted performance")
        
        recommendations.append("Regularly monitor and rebalance portfolio to maintain target risk levels")
        recommendations.append("Consider diversifying across different asset classes and strategies")
        
        return recommendations
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management system summary."""
        
        return {
            'system_status': {
                'models_initialized': len(self.historical_data) > 0,
                'assets_tracked': len(self.historical_data),
                'correlation_matrix_size': self.correlation_matrix.shape if not self.correlation_matrix.empty else (0, 0),
                'risk_factors_available': hasattr(self, 'risk_factors')
            },
            'supported_metrics': [
                'Value at Risk (VaR) - Historical, Parametric, Monte Carlo, EWMA',
                'Expected Shortfall (CVaR)',
                'Maximum Drawdown',
                'Sharpe, Sortino, and Calmar Ratios',
                'Beta and Correlation Analysis',
                'Stress Testing',
                'Monte Carlo Simulation',
                'Fat Tail and Skewness Analysis'
            ],
            'model_parameters': {
                'confidence_levels': self.confidence_levels,
                'lookback_periods': self.lookback_periods,
                'monte_carlo_simulations': self.mc_simulations,
                'risk_free_rate': self.risk_free_rate
            },
            'available_assets': list(self.historical_data.keys()),
            'last_updated': datetime.now().isoformat()
        }

# Global risk models instance
risk_models = AdvancedRiskModels()