import asyncio
import aiohttp
from web3 import Web3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging

from src.utils.logging_config import crypto_logger

@dataclass
class WhaleTransaction:
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    value: float  # In ETH or token amount
    value_usd: float
    token_symbol: str
    transaction_type: str  # 'transfer', 'swap', 'deposit', 'withdrawal'
    exchange: Optional[str]
    gas_used: int
    gas_price: float
    whale_category: str  # 'mega_whale', 'whale', 'large_holder'

@dataclass
class WhaleWallet:
    address: str
    label: Optional[str]
    balance_eth: float
    balance_usd: float
    token_holdings: Dict[str, float]
    first_seen: datetime
    last_activity: datetime
    total_transactions: int
    net_flow_24h: float
    net_flow_7d: float
    risk_score: float  # 0-100
    whale_tier: str
    exchange_connections: List[str]
    defi_interactions: List[str]

@dataclass
class MarketImpactEvent:
    event_id: str
    timestamp: datetime
    event_type: str  # 'large_transfer', 'exchange_inflow', 'exchange_outflow', 'defi_interaction'
    whale_address: str
    amount_moved: float
    token_symbol: str
    before_price: float
    after_price: float
    price_impact: float
    volume_spike: bool
    market_reaction_score: float

class OnChainWhaleTracker:
    """Advanced on-chain analysis and whale wallet tracking system."""
    
    def __init__(self):
        self.web3_connections = {}
        self.whale_wallets = {}
        self.transaction_cache = {}
        self.market_impact_events = []
        
        # Whale classification thresholds (in USD)
        self.whale_thresholds = {
            'mega_whale': 100_000_000,  # $100M+
            'whale': 10_000_000,        # $10M+
            'large_holder': 1_000_000,  # $1M+
            'significant': 100_000      # $100K+
        }
        
        # Known whale addresses and labels
        self.known_whales = {
            # Exchanges
            '0x28C6c06298d514Db089934071355E5743bf21d60': 'Binance Hot Wallet',
            '0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549': 'Binance Cold Wallet',
            '0x503828976D22510aad0201ac7EC88293211D23Da': 'Coinbase Wallet',
            '0x71660c4005BA85c37ccec55d0C4493E66Fe775d3': 'Coinbase Cold Storage',
            
            # DeFi Protocols
            '0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed': 'Aave Pool',
            '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9': 'Aave Lending Pool',
            '0xA0b86a33E6441547A5CE1A2b3CAd2bC73c7c7dE6': 'Compound cETH',
            
            # Known Whales
            '0x742d35Cc6635Cb0532043E9c2B93F8c3C2B1d11f': 'Ethereum Foundation',
            '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8': 'Tesla Bitcoin Address',
        }
        
        # RPC endpoints for different networks
        self.rpc_endpoints = {
            'ethereum': [
                'https://eth-mainnet.alchemyapi.io/v2/demo',
                'https://mainnet.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161',
                'https://ethereum.publicnode.com'
            ],
            'bsc': [
                'https://bsc-dataseed.binance.org',
                'https://bsc-dataseed1.defibit.io'
            ],
            'polygon': [
                'https://polygon-rpc.com',
                'https://rpc-mainnet.maticvigil.com'
            ]
        }
        
        # API endpoints for additional data
        self.api_endpoints = {
            'etherscan': 'https://api.etherscan.io/api',
            'moralis': 'https://deep-index.moralis.io/api/v2',
            'covalent': 'https://api.covalenthq.com/v1',
            'dune': 'https://api.dune.com/api/v1'
        }
    
    async def initialize_connections(self):
        """Initialize blockchain connections."""
        crypto_logger.logger.info("Initializing blockchain connections for whale tracking")
        
        try:
            # Initialize Web3 connections for multiple networks
            for network, endpoints in self.rpc_endpoints.items():
                for endpoint in endpoints:
                    try:
                        w3 = Web3(Web3.HTTPProvider(endpoint))
                        if w3.is_connected():
                            self.web3_connections[network] = w3
                            crypto_logger.logger.info(f"✓ Connected to {network.capitalize()} via {endpoint}")
                            break
                    except Exception as e:
                        crypto_logger.logger.debug(f"Failed to connect to {network} via {endpoint}: {e}")
            
            crypto_logger.logger.info(f"Successfully connected to {len(self.web3_connections)} blockchain networks")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing blockchain connections: {e}")
    
    async def track_large_transactions(self, min_value_usd: float = 1_000_000, 
                                     last_blocks: int = 100) -> List[WhaleTransaction]:
        """Track large transactions across monitored addresses."""
        crypto_logger.logger.info(f"Tracking transactions > ${min_value_usd:,.0f} in last {last_blocks} blocks")
        
        large_transactions = []
        
        try:
            if 'ethereum' in self.web3_connections:
                w3 = self.web3_connections['ethereum']
                
                # Get latest block number
                latest_block = w3.eth.get_block('latest')
                start_block = latest_block['number'] - last_blocks
                
                # Monitor recent blocks for large transactions
                for block_num in range(start_block, latest_block['number'] + 1):
                    try:
                        block = w3.eth.get_block(block_num, full_transactions=True)
                        
                        for tx in block['transactions']:
                            if tx['value'] > 0:
                                # Convert Wei to ETH
                                eth_amount = w3.from_wei(tx['value'], 'ether')
                                
                                # Estimate USD value (would use real ETH price)
                                eth_price_usd = 2000  # Mock price - would fetch real price
                                usd_value = float(eth_amount) * eth_price_usd
                                
                                if usd_value >= min_value_usd:
                                    # Classify transaction
                                    tx_type = self._classify_transaction(tx)
                                    whale_category = self._classify_whale_by_value(usd_value)
                                    
                                    # Get transaction receipt for gas details
                                    receipt = w3.eth.get_transaction_receipt(tx['hash'])
                                    
                                    large_tx = WhaleTransaction(
                                        tx_hash=tx['hash'].hex(),
                                        block_number=block_num,
                                        timestamp=datetime.fromtimestamp(block['timestamp']),
                                        from_address=tx['from'],
                                        to_address=tx['to'] if tx['to'] else 'Contract Creation',
                                        value=float(eth_amount),
                                        value_usd=usd_value,
                                        token_symbol='ETH',
                                        transaction_type=tx_type,
                                        exchange=self._identify_exchange(tx['to']),
                                        gas_used=receipt['gasUsed'],
                                        gas_price=float(w3.from_wei(tx['gasPrice'], 'gwei')),
                                        whale_category=whale_category
                                    )
                                    
                                    large_transactions.append(large_tx)
                                    
                                    crypto_logger.logger.info(
                                        f"🐋 Large transaction detected: {usd_value:,.0f} USD - "
                                        f"{tx['hash'].hex()[:10]}..."
                                    )
                    
                    except Exception as e:
                        crypto_logger.logger.debug(f"Error processing block {block_num}: {e}")
        
        except Exception as e:
            crypto_logger.logger.error(f"Error tracking large transactions: {e}")
        
        return large_transactions
    
    async def analyze_whale_wallets(self, addresses: List[str]) -> Dict[str, WhaleWallet]:
        """Analyze whale wallet holdings and activity patterns."""
        crypto_logger.logger.info(f"Analyzing {len(addresses)} whale wallets")
        
        whale_analysis = {}
        
        try:
            if 'ethereum' not in self.web3_connections:
                return whale_analysis
            
            w3 = self.web3_connections['ethereum']
            
            async with aiohttp.ClientSession() as session:
                for address in addresses:
                    try:
                        # Get ETH balance
                        balance_wei = w3.eth.get_balance(address)
                        balance_eth = w3.from_wei(balance_wei, 'ether')
                        balance_usd = float(balance_eth) * 2000  # Mock ETH price
                        
                        # Get transaction count
                        tx_count = w3.eth.get_transaction_count(address)
                        
                        # Get token holdings (using external API)
                        token_holdings = await self._get_token_holdings(session, address)
                        
                        # Analyze transaction history
                        tx_analysis = await self._analyze_transaction_history(session, address)
                        
                        # Calculate risk score
                        risk_score = self._calculate_wallet_risk_score(
                            balance_usd, tx_analysis, token_holdings
                        )
                        
                        # Classify whale tier
                        whale_tier = self._classify_whale_by_value(balance_usd)
                        
                        whale_wallet = WhaleWallet(
                            address=address,
                            label=self.known_whales.get(address),
                            balance_eth=float(balance_eth),
                            balance_usd=balance_usd,
                            token_holdings=token_holdings,
                            first_seen=tx_analysis.get('first_tx_date', datetime.now()),
                            last_activity=tx_analysis.get('last_tx_date', datetime.now()),
                            total_transactions=tx_count,
                            net_flow_24h=tx_analysis.get('net_flow_24h', 0),
                            net_flow_7d=tx_analysis.get('net_flow_7d', 0),
                            risk_score=risk_score,
                            whale_tier=whale_tier,
                            exchange_connections=tx_analysis.get('exchange_interactions', []),
                            defi_interactions=tx_analysis.get('defi_interactions', [])
                        )
                        
                        whale_analysis[address] = whale_wallet
                        crypto_logger.logger.info(f"✓ Analyzed whale wallet: {address[:10]}...")
                        
                    except Exception as e:
                        crypto_logger.logger.error(f"Error analyzing wallet {address}: {e}")
        
        except Exception as e:
            crypto_logger.logger.error(f"Error in whale wallet analysis: {e}")
        
        return whale_analysis
    
    async def detect_market_impact_events(self, time_window_hours: int = 24) -> List[MarketImpactEvent]:
        """Detect whale movements that potentially impact market prices."""
        crypto_logger.logger.info(f"Detecting market impact events in last {time_window_hours}h")
        
        impact_events = []
        
        try:
            # Get large transactions from recent period
            large_transactions = await self.track_large_transactions(
                min_value_usd=5_000_000,  # $5M+ for market impact analysis
                last_blocks=int(time_window_hours * 300)  # ~12s per block
            )
            
            # Analyze each transaction for market impact
            for tx in large_transactions:
                # Mock price impact analysis (would use real price data)
                price_impact = await self._analyze_price_impact(tx)
                
                if price_impact and abs(price_impact.get('impact_pct', 0)) > 2:  # >2% price impact
                    impact_event = MarketImpactEvent(
                        event_id=f"{tx.tx_hash[:16]}_{int(tx.timestamp.timestamp())}",
                        timestamp=tx.timestamp,
                        event_type=tx.transaction_type,
                        whale_address=tx.from_address,
                        amount_moved=tx.value_usd,
                        token_symbol=tx.token_symbol,
                        before_price=price_impact.get('before_price', 0),
                        after_price=price_impact.get('after_price', 0),
                        price_impact=price_impact.get('impact_pct', 0),
                        volume_spike=price_impact.get('volume_spike', False),
                        market_reaction_score=price_impact.get('reaction_score', 0)
                    )
                    
                    impact_events.append(impact_event)
                    
                    crypto_logger.logger.info(
                        f"📊 Market impact detected: {tx.value_usd:,.0f} USD movement "
                        f"caused {price_impact['impact_pct']:.1f}% price change"
                    )
        
        except Exception as e:
            crypto_logger.logger.error(f"Error detecting market impact events: {e}")
        
        # Sort by market reaction score
        impact_events.sort(key=lambda x: x.market_reaction_score, reverse=True)
        return impact_events
    
    async def _get_token_holdings(self, session: aiohttp.ClientSession, address: str) -> Dict[str, float]:
        """Get ERC-20 token holdings for an address."""
        token_holdings = {}
        
        try:
            # Mock token holdings (would use real API like Moralis, Covalent, etc.)
            # This would fetch actual token balances from blockchain APIs
            
            # Example structure - would be replaced with real API calls
            mock_holdings = {
                'USDT': np.random.uniform(100000, 10000000),
                'USDC': np.random.uniform(50000, 5000000),
                'WETH': np.random.uniform(100, 1000),
                'UNI': np.random.uniform(1000, 100000),
                'AAVE': np.random.uniform(500, 50000)
            }
            
            # Filter out small holdings
            token_holdings = {
                token: amount for token, amount in mock_holdings.items()
                if amount > 1000  # Only include significant holdings
            }
            
        except Exception as e:
            crypto_logger.logger.debug(f"Error getting token holdings for {address}: {e}")
        
        return token_holdings
    
    async def _analyze_transaction_history(self, session: aiohttp.ClientSession, address: str) -> Dict[str, Any]:
        """Analyze transaction history patterns for a wallet."""
        analysis = {
            'first_tx_date': datetime.now() - timedelta(days=365),
            'last_tx_date': datetime.now(),
            'net_flow_24h': 0,
            'net_flow_7d': 0,
            'exchange_interactions': [],
            'defi_interactions': [],
            'transaction_patterns': {}
        }
        
        try:
            # Mock transaction analysis (would use real blockchain data)
            # This would analyze actual transaction history from APIs
            
            # Simulate exchange interactions
            exchanges = ['Binance', 'Coinbase', 'Kraken', 'FTX']
            analysis['exchange_interactions'] = np.random.choice(exchanges, size=2, replace=False).tolist()
            
            # Simulate DeFi interactions
            defi_protocols = ['Uniswap', 'Aave', 'Compound', 'Curve', 'Yearn']
            analysis['defi_interactions'] = np.random.choice(defi_protocols, size=3, replace=False).tolist()
            
            # Simulate net flows
            analysis['net_flow_24h'] = np.random.uniform(-1000000, 1000000)
            analysis['net_flow_7d'] = np.random.uniform(-5000000, 5000000)
            
        except Exception as e:
            crypto_logger.logger.debug(f"Error analyzing transaction history for {address}: {e}")
        
        return analysis
    
    async def _analyze_price_impact(self, transaction: WhaleTransaction) -> Optional[Dict[str, Any]]:
        """Analyze potential price impact of a large transaction."""
        try:
            # Mock price impact analysis (would use real price data and volume)
            # This would correlate transaction timing with price movements
            
            if transaction.value_usd > 10_000_000:  # $10M+ transactions
                # Simulate price impact
                base_impact = min(transaction.value_usd / 100_000_000, 0.1)  # Max 10% impact
                random_factor = np.random.uniform(0.5, 2.0)
                impact_pct = base_impact * random_factor * np.random.choice([-1, 1])
                
                return {
                    'before_price': 2000.0,  # Mock ETH price before
                    'after_price': 2000.0 * (1 + impact_pct),
                    'impact_pct': impact_pct * 100,
                    'volume_spike': abs(impact_pct) > 0.03,  # >3% impact = volume spike
                    'reaction_score': min(abs(impact_pct) * 100, 100)
                }
            
            return None
            
        except Exception as e:
            crypto_logger.logger.debug(f"Error analyzing price impact: {e}")
            return None
    
    def _classify_transaction(self, transaction: Dict) -> str:
        """Classify transaction type based on to/from addresses."""
        to_address = transaction.get('to', '')
        
        if to_address in self.known_whales:
            label = self.known_whales[to_address]
            if 'exchange' in label.lower() or 'binance' in label.lower() or 'coinbase' in label.lower():
                return 'exchange_deposit'
            elif 'aave' in label.lower() or 'compound' in label.lower():
                return 'defi_interaction'
        
        # Default classification
        if transaction.get('input', '0x') != '0x':
            return 'contract_interaction'
        else:
            return 'transfer'
    
    def _classify_whale_by_value(self, usd_value: float) -> str:
        """Classify whale category by USD value."""
        if usd_value >= self.whale_thresholds['mega_whale']:
            return 'mega_whale'
        elif usd_value >= self.whale_thresholds['whale']:
            return 'whale'
        elif usd_value >= self.whale_thresholds['large_holder']:
            return 'large_holder'
        else:
            return 'significant'
    
    def _identify_exchange(self, address: Optional[str]) -> Optional[str]:
        """Identify if address belongs to a known exchange."""
        if not address:
            return None
        
        exchange_keywords = {
            'binance': 'Binance',
            'coinbase': 'Coinbase',
            'kraken': 'Kraken',
            'ftx': 'FTX',
            'okex': 'OKX'
        }
        
        address_label = self.known_whales.get(address, '').lower()
        
        for keyword, exchange_name in exchange_keywords.items():
            if keyword in address_label:
                return exchange_name
        
        return None
    
    def _calculate_wallet_risk_score(self, balance_usd: float, tx_analysis: Dict, 
                                   token_holdings: Dict) -> float:
        """Calculate risk score for a whale wallet."""
        risk_score = 0
        
        # Size risk (larger wallets = higher systemic risk)
        if balance_usd > 100_000_000:  # $100M+
            risk_score += 40
        elif balance_usd > 50_000_000:  # $50M+
            risk_score += 30
        elif balance_usd > 10_000_000:  # $10M+
            risk_score += 20
        
        # Activity risk (high turnover = higher risk)
        net_flow_7d = abs(tx_analysis.get('net_flow_7d', 0))
        if net_flow_7d > balance_usd * 0.5:  # >50% of balance moved in 7 days
            risk_score += 30
        elif net_flow_7d > balance_usd * 0.2:  # >20% moved
            risk_score += 20
        
        # Diversification risk (concentrated holdings = higher risk)
        if len(token_holdings) < 3:
            risk_score += 15
        elif len(token_holdings) < 5:
            risk_score += 10
        
        # Exchange connection risk (more exchanges = potentially higher volatility)
        exchange_count = len(tx_analysis.get('exchange_interactions', []))
        if exchange_count > 5:
            risk_score += 15
        elif exchange_count > 3:
            risk_score += 10
        
        return min(risk_score, 100)  # Cap at 100
    
    async def monitor_whale_movements(self, callback_func=None) -> None:
        """Continuously monitor whale movements and alert on significant activity."""
        crypto_logger.logger.info("Starting continuous whale movement monitoring")
        
        try:
            while True:
                # Track large transactions in recent blocks
                large_transactions = await self.track_large_transactions(
                    min_value_usd=2_000_000,  # $2M+ threshold for monitoring
                    last_blocks=50  # Monitor last 50 blocks (~10 minutes)
                )
                
                # Detect market impact events
                impact_events = await self.detect_market_impact_events(time_window_hours=1)
                
                # Alert on significant movements
                for tx in large_transactions:
                    if tx.value_usd > 10_000_000:  # $10M+ transactions
                        crypto_logger.logger.warning(
                            f"🚨 MEGA WHALE ALERT: ${tx.value_usd:,.0f} moved - "
                            f"From: {tx.from_address[:10]}... To: {tx.to_address[:10]}..."
                        )
                        
                        if callback_func:
                            await callback_func({
                                'type': 'whale_movement',
                                'transaction': tx,
                                'severity': 'high'
                            })
                
                # Alert on market impact events
                for event in impact_events:
                    if abs(event.price_impact) > 5:  # >5% price impact
                        crypto_logger.logger.warning(
                            f"📈 MARKET IMPACT: Whale movement caused {event.price_impact:.1f}% price change"
                        )
                        
                        if callback_func:
                            await callback_func({
                                'type': 'market_impact',
                                'event': event,
                                'severity': 'high' if abs(event.price_impact) > 10 else 'medium'
                            })
                
                # Wait before next monitoring cycle
                await asyncio.sleep(120)  # Check every 2 minutes
                
        except Exception as e:
            crypto_logger.logger.error(f"Error in whale movement monitoring: {e}")
    
    def generate_whale_report(self, whale_wallets: Dict[str, WhaleWallet], 
                            transactions: List[WhaleTransaction],
                            impact_events: List[MarketImpactEvent]) -> Dict[str, Any]:
        """Generate comprehensive whale activity report."""
        
        if not any([whale_wallets, transactions, impact_events]):
            return {}
        
        # Aggregate statistics
        total_whale_value = sum(whale.balance_usd for whale in whale_wallets.values())
        total_transaction_volume = sum(tx.value_usd for tx in transactions)
        
        # Risk analysis
        high_risk_wallets = [
            wallet for wallet in whale_wallets.values() 
            if wallet.risk_score > 70
        ]
        
        # Market impact analysis
        significant_impacts = [
            event for event in impact_events 
            if abs(event.price_impact) > 3
        ]
        
        # Whale tier distribution
        tier_distribution = {}
        for whale in whale_wallets.values():
            tier_distribution[whale.whale_tier] = tier_distribution.get(whale.whale_tier, 0) + 1
        
        return {
            'summary': {
                'total_whales_tracked': len(whale_wallets),
                'total_whale_value_usd': total_whale_value,
                'total_transaction_volume_24h': total_transaction_volume,
                'high_risk_wallets': len(high_risk_wallets),
                'significant_market_impacts': len(significant_impacts),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'whale_distribution': {
                'by_tier': tier_distribution,
                'by_risk_level': {
                    'high': len([w for w in whale_wallets.values() if w.risk_score > 70]),
                    'medium': len([w for w in whale_wallets.values() if 40 <= w.risk_score <= 70]),
                    'low': len([w for w in whale_wallets.values() if w.risk_score < 40])
                }
            },
            'recent_activity': {
                'large_transactions_24h': len(transactions),
                'total_volume_moved': total_transaction_volume,
                'exchange_flows': self._analyze_exchange_flows(transactions),
                'defi_interactions': self._analyze_defi_flows(transactions)
            },
            'market_impact': {
                'total_impact_events': len(impact_events),
                'significant_impacts': len(significant_impacts),
                'max_price_impact': max([abs(e.price_impact) for e in impact_events]) if impact_events else 0,
                'avg_impact': np.mean([abs(e.price_impact) for e in impact_events]) if impact_events else 0
            },
            'alerts': {
                'high_risk_wallets': [
                    {
                        'address': whale.address[:10] + '...',
                        'label': whale.label,
                        'risk_score': whale.risk_score,
                        'balance_usd': whale.balance_usd,
                        'net_flow_24h': whale.net_flow_24h
                    } for whale in high_risk_wallets[:10]  # Top 10 high-risk
                ],
                'recent_mega_moves': [
                    {
                        'tx_hash': tx.tx_hash[:16] + '...',
                        'value_usd': tx.value_usd,
                        'from': tx.from_address[:10] + '...',
                        'to': tx.to_address[:10] + '...',
                        'timestamp': tx.timestamp.isoformat()
                    } for tx in sorted(transactions, key=lambda x: x.value_usd, reverse=True)[:5]
                ]
            },
            'methodology': {
                'tracking_threshold_usd': 1_000_000,
                'networks_monitored': list(self.web3_connections.keys()),
                'data_sources': ['blockchain_direct', 'known_addresses', 'pattern_analysis'],
                'update_frequency': 'real_time'
            }
        }
    
    def _analyze_exchange_flows(self, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Analyze flows to/from exchanges."""
        inflows = sum(tx.value_usd for tx in transactions if tx.transaction_type == 'exchange_deposit')
        outflows = sum(tx.value_usd for tx in transactions if tx.transaction_type == 'exchange_withdrawal')
        
        return {
            'total_inflows': inflows,
            'total_outflows': outflows,
            'net_flow': inflows - outflows,
            'inflow_transactions': len([tx for tx in transactions if tx.transaction_type == 'exchange_deposit']),
            'outflow_transactions': len([tx for tx in transactions if tx.transaction_type == 'exchange_withdrawal'])
        }
    
    def _analyze_defi_flows(self, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Analyze DeFi protocol interactions."""
        defi_volume = sum(tx.value_usd for tx in transactions if tx.transaction_type == 'defi_interaction')
        defi_transactions = [tx for tx in transactions if tx.transaction_type == 'defi_interaction']
        
        return {
            'total_defi_volume': defi_volume,
            'defi_transactions': len(defi_transactions),
            'avg_defi_transaction_size': defi_volume / len(defi_transactions) if defi_transactions else 0
        }

# Global whale tracker instance
whale_tracker = OnChainWhaleTracker()