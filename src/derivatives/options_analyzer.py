import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import optimize
from scipy.stats import norm
import math

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class OptionData:
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid_price: float
    ask_price: float
    mid_price: float
    volume: float
    open_interest: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    intrinsic_value: float
    time_value: float
    moneyness: float
    last_updated: datetime

@dataclass
class FuturesData:
    symbol: str
    underlying: str
    expiration: datetime
    bid_price: float
    ask_price: float
    mid_price: float
    spot_price: float
    basis: float
    volume: float
    open_interest: float
    funding_rate: Optional[float]
    basis_yield: float
    contango_backwardation: str
    last_updated: datetime

@dataclass
class VolatilitySurface:
    underlying: str
    strikes: List[float]
    expirations: List[datetime]
    implied_vols: np.ndarray
    atm_vol: float
    vol_smile: Dict[float, float]
    term_structure: Dict[datetime, float]
    skew: float
    last_updated: datetime

class OptionsAnalyzer:
    """Advanced options and derivatives analysis engine."""
    
    def __init__(self):
        self.config = Config()
        self.exchanges = {}
        self.option_chains = {}
        self.futures_data = {}
        self.volatility_surfaces = {}
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
        # Options exchanges and their supported underlyings
        self.options_exchanges = {
            'deribit': {
                'supports_options': True,
                'supports_futures': True,
                'underlyings': ['BTC', 'ETH', 'SOL', 'MATIC'],
                'typical_fee': 0.0003,
                'min_premium': 0.0005
            },
            'okx': {
                'supports_options': True,
                'supports_futures': True,
                'underlyings': ['BTC', 'ETH'],
                'typical_fee': 0.0002,
                'min_premium': 0.001
            },
            'bybit': {
                'supports_options': False,
                'supports_futures': True,
                'underlyings': ['BTC', 'ETH', 'SOL', 'ADA'],
                'typical_fee': 0.0001,
                'min_premium': 0.0005
            }
        }
        
    async def initialize_derivatives_data(self):
        """Initialize derivatives data connections."""
        crypto_logger.logger.info("Initializing derivatives analysis engine")
        
        try:
            # Initialize derivatives exchanges (paper trading mode)
            self._initialize_paper_derivatives()
            
            # Initialize volatility models
            await self._initialize_volatility_models()
            
            # Load current market data
            await self._load_derivatives_data()
            
            crypto_logger.logger.info("✓ Derivatives analysis engine initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing derivatives analysis: {e}")
    
    def _initialize_paper_derivatives(self):
        """Initialize paper trading mode for derivatives."""
        crypto_logger.logger.info("Initializing paper derivatives trading mode")
        
        # Mock derivatives data for testing
        self.paper_mode = True
        self._generate_mock_options_chains()
        self._generate_mock_futures_data()
        
    def _generate_mock_options_chains(self):
        """Generate mock options chains for testing."""
        
        underlyings = [
            {'symbol': 'BTC', 'price': 45000, 'vol': 0.8},
            {'symbol': 'ETH', 'price': 2500, 'vol': 0.9}
        ]
        
        for underlying in underlyings:
            symbol = underlying['symbol']
            spot_price = underlying['price']
            base_vol = underlying['vol']
            
            options = []
            
            # Generate options for different expiries
            for days_to_expiry in [7, 14, 30, 60, 90]:
                expiry = datetime.now() + timedelta(days=days_to_expiry)
                time_to_expiry = days_to_expiry / 365.0
                
                # Generate strikes around ATM
                atm_strike = round(spot_price / 100) * 100  # Round to nearest 100
                strikes = np.arange(atm_strike * 0.7, atm_strike * 1.3, atm_strike * 0.05)
                
                for strike in strikes:
                    # Calculate implied volatility with skew
                    moneyness = strike / spot_price
                    vol_adjustment = 0.1 * (moneyness - 1) ** 2  # Volatility smile
                    implied_vol = base_vol + vol_adjustment
                    
                    # Calculate Greeks using Black-Scholes
                    for option_type in ['call', 'put']:
                        greeks = self._calculate_black_scholes_greeks(
                            spot_price, strike, time_to_expiry, self.risk_free_rate, implied_vol, option_type
                        )
                        
                        option = OptionData(
                            symbol=f"{symbol}-{expiry.strftime('%d%b%y')}-{int(strike)}-{option_type[0].upper()}",
                            underlying=symbol,
                            strike=strike,
                            expiration=expiry,
                            option_type=option_type,
                            bid_price=greeks['price'] * 0.98,
                            ask_price=greeks['price'] * 1.02,
                            mid_price=greeks['price'],
                            volume=np.random.uniform(10, 1000),
                            open_interest=np.random.uniform(100, 10000),
                            implied_volatility=implied_vol,
                            delta=greeks['delta'],
                            gamma=greeks['gamma'],
                            theta=greeks['theta'],
                            vega=greeks['vega'],
                            rho=greeks['rho'],
                            intrinsic_value=max(0, (spot_price - strike) if option_type == 'call' else (strike - spot_price)),
                            time_value=greeks['price'] - max(0, (spot_price - strike) if option_type == 'call' else (strike - spot_price)),
                            moneyness=moneyness,
                            last_updated=datetime.now()
                        )
                        
                        options.append(option)
            
            self.option_chains[symbol] = options
            
        crypto_logger.logger.info(f"Generated mock options chains for {len(underlyings)} underlyings")
    
    def _generate_mock_futures_data(self):
        """Generate mock futures data."""
        
        underlyings = [
            {'symbol': 'BTC', 'price': 45000},
            {'symbol': 'ETH', 'price': 2500}
        ]
        
        for underlying in underlyings:
            symbol = underlying['symbol']
            spot_price = underlying['price']
            
            futures = []
            
            # Generate futures for different expiries
            for months_ahead in [1, 3, 6, 12]:
                expiry = datetime.now() + timedelta(days=months_ahead * 30)
                time_to_expiry = months_ahead * 30 / 365.0
                
                # Calculate futures price with contango/backwardation
                convenience_yield = 0.02  # 2% convenience yield
                futures_price = spot_price * math.exp((self.risk_free_rate - convenience_yield) * time_to_expiry)
                
                # Add some random variation
                futures_price *= np.random.uniform(0.99, 1.01)
                
                basis = futures_price - spot_price
                basis_yield = (basis / spot_price) * (365 / (months_ahead * 30))
                
                future = FuturesData(
                    symbol=f"{symbol}-{expiry.strftime('%d%b%y')}",
                    underlying=symbol,
                    expiration=expiry,
                    bid_price=futures_price * 0.9995,
                    ask_price=futures_price * 1.0005,
                    mid_price=futures_price,
                    spot_price=spot_price,
                    basis=basis,
                    volume=np.random.uniform(1000, 50000),
                    open_interest=np.random.uniform(10000, 500000),
                    funding_rate=np.random.uniform(-0.01, 0.01) if months_ahead == 1 else None,
                    basis_yield=basis_yield,
                    contango_backwardation='contango' if futures_price > spot_price else 'backwardation',
                    last_updated=datetime.now()
                )
                
                futures.append(future)
            
            self.futures_data[symbol] = futures
            
        crypto_logger.logger.info(f"Generated mock futures data for {len(underlyings)} underlyings")
    
    async def _initialize_volatility_models(self):
        """Initialize volatility surface models."""
        
        for underlying in ['BTC', 'ETH']:
            options = self.option_chains.get(underlying, [])
            if not options:
                continue
                
            # Group options by expiry and strike
            strikes = sorted(list(set(opt.strike for opt in options)))
            expiries = sorted(list(set(opt.expiration for opt in options)))
            
            # Build volatility surface
            vol_matrix = np.zeros((len(strikes), len(expiries)))
            
            for i, strike in enumerate(strikes):
                for j, expiry in enumerate(expiries):
                    # Find option with matching strike and expiry
                    matching_option = next(
                        (opt for opt in options if opt.strike == strike and opt.expiration == expiry and opt.option_type == 'call'),
                        None
                    )
                    
                    if matching_option:
                        vol_matrix[i, j] = matching_option.implied_volatility
                    else:
                        # Interpolate or use ATM vol
                        vol_matrix[i, j] = 0.8  # Default vol
            
            # Calculate ATM volatility
            spot_price = 45000 if underlying == 'BTC' else 2500
            atm_strikes = [s for s in strikes if abs(s - spot_price) < spot_price * 0.05]
            atm_vol = np.mean([opt.implied_volatility for opt in options if opt.strike in atm_strikes and opt.option_type == 'call'])
            
            # Calculate volatility smile (30-day options)
            nearest_expiry = min(expiries, key=lambda x: abs((x - datetime.now()).days - 30))
            smile_options = [opt for opt in options if opt.expiration == nearest_expiry and opt.option_type == 'call']
            vol_smile = {opt.strike / spot_price: opt.implied_volatility for opt in smile_options}
            
            # Calculate term structure (ATM options)
            term_structure = {}
            for expiry in expiries:
                atm_option = min(
                    [opt for opt in options if opt.expiration == expiry and opt.option_type == 'call'],
                    key=lambda x: abs(x.strike - spot_price),
                    default=None
                )
                if atm_option:
                    term_structure[expiry] = atm_option.implied_volatility
            
            # Calculate skew (25-delta put vol - 25-delta call vol)
            skew = 0.05  # Mock skew value
            
            volatility_surface = VolatilitySurface(
                underlying=underlying,
                strikes=strikes,
                expirations=expiries,
                implied_vols=vol_matrix,
                atm_vol=atm_vol,
                vol_smile=vol_smile,
                term_structure=term_structure,
                skew=skew,
                last_updated=datetime.now()
            )
            
            self.volatility_surfaces[underlying] = volatility_surface
            
        crypto_logger.logger.info(f"Initialized volatility surfaces for {len(self.volatility_surfaces)} underlyings")
    
    async def _load_derivatives_data(self):
        """Load current derivatives market data."""
        # In paper mode, data is already generated
        if hasattr(self, 'paper_mode'):
            crypto_logger.logger.info("Using mock derivatives data in paper mode")
            return
        
        # Real data loading would go here
        # This would involve connecting to exchanges and fetching option chains, futures data, etc.
        pass
    
    def _calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate Black-Scholes price and Greeks."""
        
        if T <= 0 or sigma <= 0:
            return {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal cumulative distribution function
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        
        # Standard normal probability density function
        n_d1 = norm.pdf(d1)
        
        if option_type.lower() == 'call':
            # Call option
            price = S * N_d1 - K * np.exp(-r * T) * N_d2
            delta = N_d1
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
            rho = K * T * np.exp(-r * T) * N_d2 / 100
        else:
            # Put option
            price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
            delta = -N_neg_d1
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_neg_d2) / 365
            rho = -K * T * np.exp(-r * T) * N_neg_d2 / 100
        
        # Greeks common to both calls and puts
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        vega = S * n_d1 * np.sqrt(T) / 100
        
        return {
            'price': max(price, 0),
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    async def analyze_options_flow(self, underlying: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze options flow and unusual activity."""
        
        options = self.option_chains.get(underlying, [])
        if not options:
            return {'error': f'No options data available for {underlying}'}
        
        # Filter options by time window (mock filtering since we don't have real timestamps)
        recent_options = options  # In real implementation, filter by trade timestamp
        
        # Calculate flow metrics
        total_call_volume = sum(opt.volume for opt in recent_options if opt.option_type == 'call')
        total_put_volume = sum(opt.volume for opt in recent_options if opt.option_type == 'put')
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        # Identify large trades (top 10% by volume)
        sorted_options = sorted(recent_options, key=lambda x: x.volume, reverse=True)
        large_trades = sorted_options[:max(1, len(sorted_options) // 10)]
        
        # Calculate options sentiment
        net_call_volume = total_call_volume - total_put_volume
        sentiment = 'bullish' if net_call_volume > 0 else 'bearish'
        
        # Identify unusual activity
        unusual_activity = []
        avg_volume = np.mean([opt.volume for opt in recent_options])
        for opt in recent_options:
            if opt.volume > avg_volume * 5:  # 5x average volume
                unusual_activity.append({
                    'symbol': opt.symbol,
                    'volume': opt.volume,
                    'volume_ratio': opt.volume / avg_volume,
                    'strike': opt.strike,
                    'expiration': opt.expiration.strftime('%Y-%m-%d'),
                    'type': opt.option_type,
                    'implied_vol': opt.implied_volatility
                })
        
        # Calculate max pain (strike with most open interest)
        oi_by_strike = {}
        for opt in recent_options:
            if opt.strike not in oi_by_strike:
                oi_by_strike[opt.strike] = 0
            oi_by_strike[opt.strike] += opt.open_interest
        
        max_pain_strike = max(oi_by_strike.keys(), key=oi_by_strike.get) if oi_by_strike else 0
        
        return {
            'underlying': underlying,
            'time_window_hours': time_window_hours,
            'flow_metrics': {
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'put_call_ratio': put_call_ratio,
                'net_call_volume': net_call_volume,
                'sentiment': sentiment
            },
            'large_trades': large_trades[:5],  # Top 5 large trades
            'unusual_activity': unusual_activity[:10],  # Top 10 unusual trades
            'max_pain_strike': max_pain_strike,
            'total_open_interest': sum(opt.open_interest for opt in recent_options),
            'timestamp': datetime.now().isoformat()
        }
    
    async def analyze_volatility_surface(self, underlying: str) -> Dict[str, Any]:
        """Analyze volatility surface and identify opportunities."""
        
        vol_surface = self.volatility_surfaces.get(underlying)
        if not vol_surface:
            return {'error': f'No volatility surface data for {underlying}'}
        
        # Analyze volatility smile
        smile_analysis = self._analyze_volatility_smile(vol_surface.vol_smile)
        
        # Analyze term structure
        term_analysis = self._analyze_term_structure(vol_surface.term_structure)
        
        # Identify volatility opportunities
        opportunities = []
        
        # Look for cheap/expensive options
        options = self.option_chains.get(underlying, [])
        for opt in options:
            historical_vol = 0.7  # Mock historical volatility
            vol_premium = opt.implied_volatility - historical_vol
            
            if vol_premium > 0.2:  # Expensive options
                opportunities.append({
                    'type': 'expensive_option',
                    'symbol': opt.symbol,
                    'strike': opt.strike,
                    'implied_vol': opt.implied_volatility,
                    'historical_vol': historical_vol,
                    'vol_premium': vol_premium,
                    'suggestion': 'Consider selling volatility'
                })
            elif vol_premium < -0.1:  # Cheap options
                opportunities.append({
                    'type': 'cheap_option',
                    'symbol': opt.symbol,
                    'strike': opt.strike,
                    'implied_vol': opt.implied_volatility,
                    'historical_vol': historical_vol,
                    'vol_premium': vol_premium,
                    'suggestion': 'Consider buying volatility'
                })
        
        return {
            'underlying': underlying,
            'atm_volatility': vol_surface.atm_vol,
            'volatility_skew': vol_surface.skew,
            'smile_analysis': smile_analysis,
            'term_structure_analysis': term_analysis,
            'volatility_opportunities': opportunities[:10],
            'surface_updated': vol_surface.last_updated.isoformat()
        }
    
    def _analyze_volatility_smile(self, vol_smile: Dict[float, float]) -> Dict[str, Any]:
        """Analyze volatility smile patterns."""
        
        if len(vol_smile) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Sort by moneyness
        sorted_smile = sorted(vol_smile.items())
        moneyness_values = [item[0] for item in sorted_smile]
        vol_values = [item[1] for item in sorted_smile]
        
        # Find ATM volatility (closest to 1.0 moneyness)
        atm_index = min(range(len(moneyness_values)), key=lambda i: abs(moneyness_values[i] - 1.0))
        atm_vol = vol_values[atm_index]
        
        # Calculate smile slope
        if len(vol_values) >= 2:
            otm_put_vol = vol_values[0] if moneyness_values[0] < 1.0 else atm_vol
            otm_call_vol = vol_values[-1] if moneyness_values[-1] > 1.0 else atm_vol
            smile_slope = otm_put_vol - otm_call_vol
        else:
            smile_slope = 0
        
        # Determine smile pattern
        if abs(smile_slope) < 0.05:
            pattern = 'flat'
        elif smile_slope > 0.1:
            pattern = 'reverse_skew'  # Puts more expensive than calls
        elif smile_slope < -0.1:
            pattern = 'forward_skew'  # Calls more expensive than puts
        else:
            pattern = 'normal_smile'
        
        return {
            'pattern': pattern,
            'atm_volatility': atm_vol,
            'smile_slope': smile_slope,
            'min_vol': min(vol_values),
            'max_vol': max(vol_values),
            'vol_range': max(vol_values) - min(vol_values)
        }
    
    def _analyze_term_structure(self, term_structure: Dict[datetime, float]) -> Dict[str, Any]:
        """Analyze volatility term structure."""
        
        if len(term_structure) < 2:
            return {'pattern': 'insufficient_data'}
        
        # Sort by expiration
        sorted_terms = sorted(term_structure.items())
        expirations = [item[0] for item in sorted_terms]
        vols = [item[1] for item in sorted_terms]
        
        # Calculate time to expiry in days
        days_to_expiry = [(exp - datetime.now()).days for exp in expirations]
        
        # Determine term structure shape
        if len(vols) >= 2:
            short_term_vol = vols[0]
            long_term_vol = vols[-1]
            slope = long_term_vol - short_term_vol
            
            if slope > 0.1:
                shape = 'upward_sloping'  # Contango
            elif slope < -0.1:
                shape = 'downward_sloping'  # Backwardation
            else:
                shape = 'flat'
        else:
            shape = 'insufficient_data'
            slope = 0
        
        return {
            'shape': shape,
            'slope': slope,
            'short_term_vol': vols[0],
            'long_term_vol': vols[-1],
            'vol_range': max(vols) - min(vols),
            'term_points': len(vols)
        }
    
    async def analyze_futures_basis(self, underlying: str) -> Dict[str, Any]:
        """Analyze futures basis and contango/backwardation."""
        
        futures = self.futures_data.get(underlying, [])
        if not futures:
            return {'error': f'No futures data available for {underlying}'}
        
        # Sort futures by expiration
        futures_sorted = sorted(futures, key=lambda x: x.expiration)
        
        # Analyze basis term structure
        basis_curve = []
        for future in futures_sorted:
            days_to_expiry = (future.expiration - datetime.now()).days
            basis_curve.append({
                'symbol': future.symbol,
                'days_to_expiry': days_to_expiry,
                'basis': future.basis,
                'basis_yield': future.basis_yield,
                'futures_price': future.mid_price,
                'spot_price': future.spot_price,
                'pattern': future.contango_backwardation
            })
        
        # Determine overall market structure
        total_basis = sum(f.basis for f in futures_sorted)
        avg_basis = total_basis / len(futures_sorted)
        
        if avg_basis > 0:
            market_structure = 'contango'
            interpretation = 'Market expects higher prices in the future'
        else:
            market_structure = 'backwardation'
            interpretation = 'Market expects lower prices in the future'
        
        # Identify arbitrage opportunities
        arbitrage_opportunities = []
        for i in range(len(futures_sorted) - 1):
            current_future = futures_sorted[i]
            next_future = futures_sorted[i + 1]
            
            # Calculate calendar spread
            spread = next_future.mid_price - current_future.mid_price
            time_diff = (next_future.expiration - current_future.expiration).days
            
            if time_diff > 0:
                spread_per_day = spread / time_diff
                
                # Identify unusual spreads
                expected_spread = current_future.spot_price * (self.risk_free_rate / 365) * time_diff
                spread_anomaly = abs(spread - expected_spread) / current_future.spot_price
                
                if spread_anomaly > 0.01:  # 1% anomaly threshold
                    arbitrage_opportunities.append({
                        'type': 'calendar_spread',
                        'long_leg': current_future.symbol,
                        'short_leg': next_future.symbol,
                        'spread': spread,
                        'expected_spread': expected_spread,
                        'anomaly_pct': spread_anomaly * 100,
                        'opportunity': 'buy_spread' if spread < expected_spread else 'sell_spread'
                    })
        
        # Calculate storage costs and convenience yield
        nearest_future = futures_sorted[0]
        storage_cost = 0.01  # Mock 1% annual storage cost
        implied_convenience_yield = (
            self.risk_free_rate - 
            math.log(nearest_future.mid_price / nearest_future.spot_price) / 
            ((nearest_future.expiration - datetime.now()).days / 365) -
            storage_cost
        )
        
        return {
            'underlying': underlying,
            'market_structure': market_structure,
            'interpretation': interpretation,
            'average_basis': avg_basis,
            'basis_curve': basis_curve,
            'arbitrage_opportunities': arbitrage_opportunities,
            'implied_convenience_yield': implied_convenience_yield,
            'storage_cost_estimate': storage_cost,
            'total_futures_tracked': len(futures_sorted),
            'timestamp': datetime.now().isoformat()
        }
    
    async def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level Greeks and risk metrics."""
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        total_value = 0
        
        position_details = []
        
        for position in positions:
            symbol = position['symbol']
            quantity = position['quantity']
            
            # Find option data
            option_data = None
            for underlying in self.option_chains:
                option_data = next(
                    (opt for opt in self.option_chains[underlying] if opt.symbol == symbol),
                    None
                )
                if option_data:
                    break
            
            if not option_data:
                continue
            
            # Calculate position Greeks
            pos_value = option_data.mid_price * quantity
            pos_delta = option_data.delta * quantity
            pos_gamma = option_data.gamma * quantity
            pos_theta = option_data.theta * quantity
            pos_vega = option_data.vega * quantity
            pos_rho = option_data.rho * quantity
            
            position_details.append({
                'symbol': symbol,
                'quantity': quantity,
                'value': pos_value,
                'delta': pos_delta,
                'gamma': pos_gamma,
                'theta': pos_theta,
                'vega': pos_vega,
                'rho': pos_rho
            })
            
            # Add to totals
            total_value += pos_value
            total_delta += pos_delta
            total_gamma += pos_gamma
            total_theta += pos_theta
            total_vega += pos_vega
            total_rho += pos_rho
        
        # Calculate risk metrics
        delta_neutral_ratio = abs(total_delta) / max(total_value, 1) * 100
        
        # Risk assessment
        risk_level = 'low'
        if delta_neutral_ratio > 20:
            risk_level = 'high'
        elif delta_neutral_ratio > 10:
            risk_level = 'medium'
        
        return {
            'portfolio_summary': {
                'total_positions': len(positions),
                'total_value': total_value,
                'portfolio_delta': total_delta,
                'portfolio_gamma': total_gamma,
                'portfolio_theta': total_theta,
                'portfolio_vega': total_vega,
                'portfolio_rho': total_rho
            },
            'risk_metrics': {
                'delta_neutral_ratio': delta_neutral_ratio,
                'risk_level': risk_level,
                'daily_pnl_theta': total_theta,  # Expected daily P&L from time decay
                'vol_sensitivity': total_vega      # P&L sensitivity to 1% vol change
            },
            'position_details': position_details,
            'hedging_suggestions': self._generate_hedging_suggestions(total_delta, total_gamma, total_vega),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_hedging_suggestions(self, delta: float, gamma: float, vega: float) -> List[Dict[str, Any]]:
        """Generate hedging suggestions based on portfolio Greeks."""
        
        suggestions = []
        
        # Delta hedging
        if abs(delta) > 100:
            hedge_amount = -delta
            suggestions.append({
                'type': 'delta_hedge',
                'action': 'buy' if hedge_amount > 0 else 'sell',
                'amount': abs(hedge_amount),
                'instrument': 'underlying',
                'reason': f'Portfolio delta of {delta:.0f} exceeds risk tolerance'
            })
        
        # Gamma hedging
        if abs(gamma) > 10:
            suggestions.append({
                'type': 'gamma_hedge',
                'action': 'buy' if gamma < 0 else 'sell',
                'instrument': 'long_dated_options',
                'reason': f'High gamma exposure ({gamma:.2f}) creates convexity risk'
            })
        
        # Vega hedging
        if abs(vega) > 1000:
            suggestions.append({
                'type': 'vega_hedge',
                'action': 'sell' if vega > 0 else 'buy',
                'instrument': 'volatility',
                'reason': f'High vega exposure ({vega:.0f}) to volatility changes'
            })
        
        return suggestions
    
    def get_derivatives_summary(self) -> Dict[str, Any]:
        """Get comprehensive derivatives market summary."""
        
        summary = {
            'market_overview': {
                'supported_underlyings': list(self.option_chains.keys()),
                'total_options': sum(len(chains) for chains in self.option_chains.values()),
                'total_futures': sum(len(futures) for futures in self.futures_data.values()),
                'volatility_surfaces': len(self.volatility_surfaces)
            },
            'volatility_summary': {},
            'futures_summary': {},
            'options_activity': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Volatility summary
        for underlying, surface in self.volatility_surfaces.items():
            summary['volatility_summary'][underlying] = {
                'atm_vol': surface.atm_vol,
                'vol_skew': surface.skew,
                'term_structure_shape': self._analyze_term_structure(surface.term_structure)['shape']
            }
        
        # Futures summary
        for underlying, futures in self.futures_data.items():
            nearest_future = min(futures, key=lambda x: x.expiration)
            summary['futures_summary'][underlying] = {
                'nearest_expiry': nearest_future.expiration.strftime('%Y-%m-%d'),
                'basis': nearest_future.basis,
                'market_structure': nearest_future.contango_backwardation,
                'total_contracts': len(futures)
            }
        
        # Options activity summary
        for underlying, options in self.option_chains.items():
            call_volume = sum(opt.volume for opt in options if opt.option_type == 'call')
            put_volume = sum(opt.volume for opt in options if opt.option_type == 'put')
            
            summary['options_activity'][underlying] = {
                'call_volume': call_volume,
                'put_volume': put_volume,
                'put_call_ratio': put_volume / call_volume if call_volume > 0 else 0,
                'total_open_interest': sum(opt.open_interest for opt in options)
            }
        
        return summary

# Global options analyzer instance
options_analyzer = OptionsAnalyzer()