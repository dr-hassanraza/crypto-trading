import asyncio
import aiohttp
import json
from web3 import Web3
from decimal import Decimal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from src.utils.logging_config import crypto_logger

@dataclass
class YieldFarm:
    protocol: str
    pool_name: str
    tokens: List[str]
    apy: float
    tvl: float  # Total Value Locked
    risk_score: float  # 0-100
    impermanent_loss_risk: float
    smart_contract_risk: float
    liquidity_risk: float
    reward_tokens: List[str]
    entry_requirements: Dict[str, Any]
    lock_period: Optional[int]  # days
    fees: Dict[str, float]
    last_updated: datetime

@dataclass
class DeFiOpportunity:
    strategy_type: str  # 'lending', 'farming', 'staking', 'arbitrage'
    protocol: str
    asset: str
    yield_pct: float
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    min_investment: float
    expected_return_1m: float
    expected_return_3m: float
    expected_return_1y: float
    risks: List[str]
    requirements: List[str]
    gas_costs: float
    complexity_score: int  # 1-10
    timestamp: datetime

@dataclass
class LiquidityPool:
    protocol: str
    pair: str
    reserves: Dict[str, float]
    fees_24h: float
    volume_24h: float
    price_impact_1k: float
    price_impact_10k: float
    il_risk_score: float  # Impermanent loss risk
    farm_apy: Optional[float]
    trading_apy: float

class DeFiProtocolAnalyzer:
    """Advanced DeFi protocol analysis and yield farming optimization."""
    
    def __init__(self):
        self.protocols = {}
        self.yield_farms = {}
        self.pools_data = {}
        self.risk_models = {}
        
        # Protocol configurations
        self.protocol_configs = {
            'uniswap_v3': {
                'type': 'dex',
                'risk_rating': 'LOW',
                'fees': {'trading': 0.003, 'gas_estimate': 0.01},
                'api_endpoint': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                'contract_addresses': {
                    'factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                    'router': '0xE592427A0AEce92De3Edee1F18E0157C05861564'
                }
            },
            'compound': {
                'type': 'lending',
                'risk_rating': 'LOW',
                'fees': {'gas_estimate': 0.005},
                'api_endpoint': 'https://api.compound.finance/api/v2',
                'contract_addresses': {
                    'comptroller': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B'
                }
            },
            'aave': {
                'type': 'lending',
                'risk_rating': 'LOW',
                'fees': {'gas_estimate': 0.006},
                'api_endpoint': 'https://aave-api-v2.aave.com',
                'contract_addresses': {
                    'lending_pool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'
                }
            },
            'curve': {
                'type': 'dex',
                'risk_rating': 'MEDIUM',
                'fees': {'trading': 0.0004, 'gas_estimate': 0.008},
                'api_endpoint': 'https://api.curve.fi/api',
                'speciality': 'stablecoins'
            },
            'yearn': {
                'type': 'vault',
                'risk_rating': 'MEDIUM',
                'fees': {'performance': 0.20, 'management': 0.02},
                'api_endpoint': 'https://api.yearn.finance/v1/chains/1/vaults/all'
            },
            'convex': {
                'type': 'yield_farming',
                'risk_rating': 'MEDIUM',
                'fees': {'gas_estimate': 0.01},
                'api_endpoint': 'https://www.convexfinance.com/api'
            }
        }
        
        # Risk assessment weights
        self.risk_weights = {
            'smart_contract': 0.3,
            'liquidity': 0.25,
            'impermanent_loss': 0.2,
            'protocol': 0.15,
            'market': 0.1
        }
        
    async def initialize_web3_connections(self):
        """Initialize Web3 connections to Ethereum mainnet."""
        try:
            # Use public RPC endpoints (in production, use private endpoints)
            rpc_endpoints = [
                'https://eth-mainnet.alchemyapi.io/v2/demo',
                'https://mainnet.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161',
                'https://ethereum.publicnode.com'
            ]
            
            for endpoint in rpc_endpoints:
                try:
                    w3 = Web3(Web3.HTTPProvider(endpoint))
                    if w3.is_connected():
                        self.w3 = w3
                        crypto_logger.logger.info(f"✓ Connected to Ethereum via {endpoint}")
                        break
                except Exception as e:
                    crypto_logger.logger.debug(f"Failed to connect to {endpoint}: {e}")
            
            if not hasattr(self, 'w3'):
                crypto_logger.logger.warning("No Web3 connection established - using API-only mode")
                
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing Web3: {e}")
    
    async def fetch_yield_opportunities(self) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch yield farming opportunities from multiple protocols."""
        crypto_logger.logger.info("Fetching DeFi yield opportunities")
        
        opportunities = {
            'lending': [],
            'farming': [],
            'staking': [],
            'vaults': []
        }
        
        async with aiohttp.ClientSession() as session:
            # Fetch from multiple protocols concurrently
            tasks = [
                self._fetch_compound_rates(session),
                self._fetch_aave_rates(session),
                self._fetch_yearn_vaults(session),
                self._fetch_curve_pools(session),
                self._fetch_uniswap_farms(session)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    for category, opps in result.items():
                        if category in opportunities:
                            opportunities[category].extend(opps)
        
        # Sort by yield and filter by risk
        for category in opportunities:
            opportunities[category].sort(key=lambda x: x.yield_pct, reverse=True)
            opportunities[category] = opportunities[category][:20]  # Top 20 per category
        
        crypto_logger.logger.info(f"Found {sum(len(opps) for opps in opportunities.values())} opportunities")
        
        return opportunities
    
    async def _fetch_compound_rates(self, session: aiohttp.ClientSession) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch lending rates from Compound."""
        opportunities = {'lending': []}
        
        try:
            url = "https://api.compound.finance/api/v2/ctoken"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for token_data in data.get('cToken', []):
                        if token_data.get('supply_rate'):
                            supply_apy = float(token_data['supply_rate']['value']) * 100
                            
                            if supply_apy > 0.1:  # Only include meaningful yields
                                opportunity = DeFiOpportunity(
                                    strategy_type='lending',
                                    protocol='compound',
                                    asset=token_data['underlying_symbol'],
                                    yield_pct=supply_apy,
                                    risk_level='LOW',
                                    min_investment=100.0,
                                    expected_return_1m=supply_apy/12,
                                    expected_return_3m=supply_apy/4,
                                    expected_return_1y=supply_apy,
                                    risks=['smart_contract', 'liquidation'],
                                    requirements=['ethereum_wallet'],
                                    gas_costs=0.005,
                                    complexity_score=2,
                                    timestamp=datetime.now()
                                )
                                opportunities['lending'].append(opportunity)
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error fetching Compound rates: {e}")
        
        return opportunities
    
    async def _fetch_aave_rates(self, session: aiohttp.ClientSession) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch lending rates from Aave."""
        opportunities = {'lending': []}
        
        try:
            url = "https://aave-api-v2.aave.com/data/reserves-incentives-mainnet"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for reserve in data:
                        supply_apy = float(reserve.get('liquidityRate', 0)) / 1e25 * 100
                        incentive_apy = float(reserve.get('aIncentivesAPY', 0))
                        total_apy = supply_apy + incentive_apy
                        
                        if total_apy > 0.1:
                            opportunity = DeFiOpportunity(
                                strategy_type='lending',
                                protocol='aave',
                                asset=reserve['symbol'],
                                yield_pct=total_apy,
                                risk_level='LOW',
                                min_investment=50.0,
                                expected_return_1m=total_apy/12,
                                expected_return_3m=total_apy/4,
                                expected_return_1y=total_apy,
                                risks=['smart_contract', 'liquidation'],
                                requirements=['ethereum_wallet'],
                                gas_costs=0.006,
                                complexity_score=2,
                                timestamp=datetime.now()
                            )
                            opportunities['lending'].append(opportunity)
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error fetching Aave rates: {e}")
        
        return opportunities
    
    async def _fetch_yearn_vaults(self, session: aiohttp.ClientSession) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch vault yields from Yearn Finance."""
        opportunities = {'vaults': []}
        
        try:
            url = "https://api.yearn.finance/v1/chains/1/vaults/all"
            async with session.get(url) as response:
                if response.status == 200:
                    vaults = await response.json()
                    
                    for vault in vaults:
                        if vault.get('apy'):
                            apy_data = vault['apy']
                            net_apy = float(apy_data.get('net_apy', 0)) * 100
                            
                            if net_apy > 0.5:
                                risk_level = self._assess_vault_risk(vault)
                                
                                opportunity = DeFiOpportunity(
                                    strategy_type='vault',
                                    protocol='yearn',
                                    asset=vault['token']['symbol'],
                                    yield_pct=net_apy,
                                    risk_level=risk_level,
                                    min_investment=float(vault.get('depositLimit', 0)) / 1e18 * 0.01,
                                    expected_return_1m=net_apy/12,
                                    expected_return_3m=net_apy/4,
                                    expected_return_1y=net_apy,
                                    risks=['smart_contract', 'strategy_risk', 'impermanent_loss'],
                                    requirements=['ethereum_wallet', 'gas_fees'],
                                    gas_costs=0.008,
                                    complexity_score=4,
                                    timestamp=datetime.now()
                                )
                                opportunities['vaults'].append(opportunity)
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error fetching Yearn vaults: {e}")
        
        return opportunities
    
    async def _fetch_curve_pools(self, session: aiohttp.ClientSession) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch Curve pool data and farming opportunities."""
        opportunities = {'farming': []}
        
        try:
            url = "https://api.curve.fi/api/getPools/ethereum/main"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pools = data.get('data', {}).get('poolData', [])
                    
                    for pool in pools:
                        if pool.get('gaugeCrvApy'):
                            base_apy = float(pool.get('latestDailyApy', 0))
                            reward_apy = float(pool['gaugeCrvApy'][0]) if pool['gaugeCrvApy'] else 0
                            total_apy = base_apy + reward_apy
                            
                            if total_apy > 1.0:
                                # Assess impermanent loss risk
                                il_risk = self._assess_curve_il_risk(pool)
                                risk_level = 'LOW' if il_risk < 0.02 else 'MEDIUM' if il_risk < 0.05 else 'HIGH'
                                
                                opportunity = DeFiOpportunity(
                                    strategy_type='farming',
                                    protocol='curve',
                                    asset=pool['name'],
                                    yield_pct=total_apy,
                                    risk_level=risk_level,
                                    min_investment=1000.0,
                                    expected_return_1m=total_apy/12,
                                    expected_return_3m=total_apy/4,
                                    expected_return_1y=total_apy,
                                    risks=['impermanent_loss', 'smart_contract', 'reward_token_risk'],
                                    requirements=['liquidity_provision', 'gauge_staking'],
                                    gas_costs=0.015,
                                    complexity_score=6,
                                    timestamp=datetime.now()
                                )
                                opportunities['farming'].append(opportunity)
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error fetching Curve pools: {e}")
        
        return opportunities
    
    async def _fetch_uniswap_farms(self, session: aiohttp.ClientSession) -> Dict[str, List[DeFiOpportunity]]:
        """Fetch Uniswap V3 farming opportunities."""
        opportunities = {'farming': []}
        
        try:
            # Query Uniswap V3 subgraph for top pools
            query = """
            {
                pools(first: 20, orderBy: volumeUSD, orderDirection: desc) {
                    id
                    token0 {
                        symbol
                        decimals
                    }
                    token1 {
                        symbol
                        decimals
                    }
                    feeTier
                    liquidity
                    volumeUSD
                    totalValueLockedUSD
                    feeGrowthGlobal0X128
                    feeGrowthGlobal1X128
                }
            }
            """
            
            url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
            async with session.post(url, json={'query': query}) as response:
                if response.status == 200:
                    data = await response.json()
                    pools = data.get('data', {}).get('pools', [])
                    
                    for pool in pools:
                        if float(pool['totalValueLockedUSD']) > 1000000:  # Min $1M TVL
                            # Calculate estimated APY from fees
                            daily_volume = float(pool['volumeUSD'])
                            tvl = float(pool['totalValueLockedUSD'])
                            fee_tier = int(pool['feeTier']) / 1000000  # Convert to decimal
                            
                            # Estimate daily fees
                            daily_fees = daily_volume * fee_tier
                            annual_fees = daily_fees * 365
                            estimated_apy = (annual_fees / tvl) * 100 if tvl > 0 else 0
                            
                            if estimated_apy > 5.0:  # Only high-yield opportunities
                                pair_name = f"{pool['token0']['symbol']}/{pool['token1']['symbol']}"
                                
                                opportunity = DeFiOpportunity(
                                    strategy_type='farming',
                                    protocol='uniswap_v3',
                                    asset=pair_name,
                                    yield_pct=estimated_apy,
                                    risk_level='MEDIUM',
                                    min_investment=5000.0,  # Higher minimum due to V3 complexity
                                    expected_return_1m=estimated_apy/12,
                                    expected_return_3m=estimated_apy/4,
                                    expected_return_1y=estimated_apy,
                                    risks=['impermanent_loss', 'range_management', 'gas_costs'],
                                    requirements=['active_management', 'advanced_knowledge'],
                                    gas_costs=0.02,  # Higher due to position management
                                    complexity_score=8,  # Very complex
                                    timestamp=datetime.now()
                                )
                                opportunities['farming'].append(opportunity)
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error fetching Uniswap farms: {e}")
        
        return opportunities
    
    def analyze_impermanent_loss(self, token_a: str, token_b: str, 
                                price_change_a: float, price_change_b: float) -> Dict[str, float]:
        """Calculate impermanent loss for a liquidity pair."""
        
        # Price ratio change
        price_ratio_change = (1 + price_change_a) / (1 + price_change_b)
        
        # Impermanent loss formula
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        il_multiplier = (2 * np.sqrt(price_ratio_change)) / (1 + price_ratio_change) - 1
        il_percentage = il_multiplier * 100
        
        # Value without IL (just holding)
        hold_value = 0.5 * (1 + price_change_a) + 0.5 * (1 + price_change_b)
        
        # Value with LP (including IL)
        lp_value = hold_value * (1 + il_multiplier)
        
        return {
            'impermanent_loss_pct': il_percentage,
            'hold_value': hold_value,
            'lp_value': lp_value,
            'il_dollar_amount': abs(il_percentage * hold_value / 100) if il_percentage < 0 else 0
        }
    
    def calculate_yield_optimization(self, available_capital: float, 
                                   risk_tolerance: str) -> Dict[str, Any]:
        """Calculate optimal yield farming strategy allocation."""
        
        # Risk tolerance mappings
        risk_mappings = {
            'conservative': {'max_risk': 'MEDIUM', 'allocation': {'LOW': 0.7, 'MEDIUM': 0.3}},
            'moderate': {'max_risk': 'HIGH', 'allocation': {'LOW': 0.4, 'MEDIUM': 0.4, 'HIGH': 0.2}},
            'aggressive': {'max_risk': 'EXTREME', 'allocation': {'LOW': 0.2, 'MEDIUM': 0.3, 'HIGH': 0.3, 'EXTREME': 0.2}}
        }
        
        risk_config = risk_mappings.get(risk_tolerance.lower(), risk_mappings['moderate'])
        
        # Mock optimization (in practice, would use actual opportunities)
        strategy_allocation = {
            'lending_stable': {
                'protocols': ['aave_usdc', 'compound_dai'],
                'allocation_pct': risk_config['allocation'].get('LOW', 0) * 100,
                'expected_apy': 4.5,
                'risk_level': 'LOW'
            },
            'lp_farming': {
                'protocols': ['curve_3pool', 'yearn_usdc'],
                'allocation_pct': risk_config['allocation'].get('MEDIUM', 0) * 100,
                'expected_apy': 12.8,
                'risk_level': 'MEDIUM'
            },
            'yield_farming': {
                'protocols': ['convex_frax', 'uniswap_v3_eth_usdc'],
                'allocation_pct': risk_config['allocation'].get('HIGH', 0) * 100,
                'expected_apy': 25.4,
                'risk_level': 'HIGH'
            }
        }
        
        # Calculate expected returns
        total_expected_apy = sum(
            strategy['allocation_pct'] / 100 * strategy['expected_apy']
            for strategy in strategy_allocation.values()
        )
        
        annual_return = available_capital * total_expected_apy / 100
        
        return {
            'total_capital': available_capital,
            'strategy_allocation': strategy_allocation,
            'expected_apy': total_expected_apy,
            'expected_annual_return': annual_return,
            'expected_monthly_return': annual_return / 12,
            'risk_score': self._calculate_portfolio_risk_score(strategy_allocation),
            'rebalance_frequency': 'monthly',
            'gas_costs_annual': 500,  # Estimated annual gas costs
            'net_return': annual_return - 500
        }
    
    def _assess_vault_risk(self, vault: Dict[str, Any]) -> str:
        """Assess risk level for Yearn vault."""
        # Simple risk assessment based on vault properties
        strategies = vault.get('strategies', [])
        if not strategies:
            return 'MEDIUM'
        
        # Check for risky strategies
        risky_keywords = ['leverage', 'convex', 'curve', 'exotic']
        safe_keywords = ['stable', 'usdc', 'dai', 'usdt']
        
        vault_name = vault.get('name', '').lower()
        
        if any(keyword in vault_name for keyword in safe_keywords):
            return 'LOW'
        elif any(keyword in vault_name for keyword in risky_keywords):
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _assess_curve_il_risk(self, pool: Dict[str, Any]) -> float:
        """Assess impermanent loss risk for Curve pool."""
        # Curve pools with similar assets have lower IL risk
        pool_name = pool.get('name', '').lower()
        
        if 'stable' in pool_name or any(stable in pool_name for stable in ['usdc', 'usdt', 'dai']):
            return 0.01  # Very low IL risk for stablecoins
        elif 'btc' in pool_name and 'eth' in pool_name:
            return 0.03  # Low-medium IL risk for correlated assets
        else:
            return 0.08  # Higher IL risk for uncorrelated assets
    
    def _calculate_portfolio_risk_score(self, allocation: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk score."""
        risk_scores = {'LOW': 20, 'MEDIUM': 50, 'HIGH': 80, 'EXTREME': 100}
        
        weighted_risk = sum(
            strategy['allocation_pct'] / 100 * risk_scores.get(strategy['risk_level'], 50)
            for strategy in allocation.values()
        )
        
        return weighted_risk
    
    async def monitor_defi_positions(self, wallet_address: str) -> Dict[str, Any]:
        """Monitor existing DeFi positions for a wallet."""
        if not hasattr(self, 'w3'):
            return {'error': 'Web3 connection not available'}
        
        positions = {
            'lending': [],
            'farming': [],
            'staking': [],
            'total_value': 0
        }
        
        try:
            # This would require specific contract interactions
            # For now, return mock data structure
            positions['summary'] = {
                'total_positions': 0,
                'total_value_usd': 0,
                'total_yield_earned_24h': 0,
                'average_apy': 0,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            crypto_logger.logger.error(f"Error monitoring DeFi positions: {e}")
        
        return positions
    
    def generate_defi_strategy_report(self, opportunities: Dict[str, List[DeFiOpportunity]], 
                                    user_capital: float, risk_tolerance: str) -> Dict[str, Any]:
        """Generate comprehensive DeFi strategy report."""
        
        # Filter opportunities by risk tolerance
        risk_filter = {
            'conservative': ['LOW'],
            'moderate': ['LOW', 'MEDIUM'],
            'aggressive': ['LOW', 'MEDIUM', 'HIGH']
        }
        
        allowed_risks = risk_filter.get(risk_tolerance.lower(), ['LOW', 'MEDIUM'])
        
        filtered_opportunities = {}
        for category, opps in opportunities.items():
            filtered_opportunities[category] = [
                opp for opp in opps 
                if opp.risk_level in allowed_risks and opp.min_investment <= user_capital * 0.3
            ]
        
        # Calculate strategy recommendations
        strategy = self.calculate_yield_optimization(user_capital, risk_tolerance)
        
        # Top recommendations
        top_recommendations = []
        for category, opps in filtered_opportunities.items():
            if opps:
                top_recommendations.extend(opps[:3])  # Top 3 from each category
        
        top_recommendations.sort(key=lambda x: x.yield_pct, reverse=True)
        
        return {
            'user_profile': {
                'capital': user_capital,
                'risk_tolerance': risk_tolerance,
                'experience_level': 'intermediate'  # Could be input parameter
            },
            'strategy_overview': strategy,
            'top_opportunities': top_recommendations[:10],
            'opportunities_by_category': filtered_opportunities,
            'market_insights': {
                'avg_lending_apy': np.mean([opp.yield_pct for opp in filtered_opportunities.get('lending', [])]) if filtered_opportunities.get('lending') else 0,
                'avg_farming_apy': np.mean([opp.yield_pct for opp in filtered_opportunities.get('farming', [])]) if filtered_opportunities.get('farming') else 0,
                'total_opportunities': sum(len(opps) for opps in filtered_opportunities.values()),
                'risk_distribution': self._analyze_risk_distribution(top_recommendations)
            },
            'gas_cost_estimates': {
                'entry_cost': 50,  # USD
                'management_monthly': 25,
                'exit_cost': 30
            },
            'next_steps': [
                'Set up Ethereum wallet with sufficient gas funds',
                'Start with lower-risk lending protocols',
                'Gradually increase exposure to higher-yield strategies',
                'Monitor and rebalance monthly'
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_risk_distribution(self, opportunities: List[DeFiOpportunity]) -> Dict[str, int]:
        """Analyze risk distribution of opportunities."""
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'EXTREME': 0}
        
        for opp in opportunities:
            risk_counts[opp.risk_level] += 1
        
        return risk_counts

# Global DeFi analyzer instance
defi_analyzer = DeFiProtocolAnalyzer()