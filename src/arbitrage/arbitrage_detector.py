import asyncio
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import defaultdict

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    volume_available: float
    potential_profit: float
    execution_time_estimate: float
    risk_score: float
    confidence: float
    timestamp: datetime
    market_conditions: Dict[str, Any]

@dataclass 
class TriangularArbitrage:
    symbol_a: str  # e.g., BTC
    symbol_b: str  # e.g., ETH  
    base_currency: str  # e.g., USDT
    exchange: str
    buy_a_with_base: float  # BTC/USDT price
    sell_a_for_b: float     # BTC/ETH price (inverted)
    sell_b_for_base: float  # ETH/USDT price
    profit_pct: float
    execution_path: List[str]
    volume_constraint: float
    timestamp: datetime

class MultiExchangeArbitrageDetector:
    """Advanced arbitrage detection across multiple exchanges."""
    
    def __init__(self):
        self.config = Config()
        self.exchanges = {}
        self.price_cache = defaultdict(dict)
        self.orderbook_cache = defaultdict(dict)
        self.fee_cache = {}
        self.latency_cache = {}
        self.min_spread_threshold = 0.5  # Minimum 0.5% spread
        self.max_execution_time = 30  # 30 seconds max execution
        self.risk_tolerance = 0.7  # Risk score threshold
        
        # Exchange configurations with API limits and fees
        self.exchange_configs = {
            'binance': {
                'fees': {'maker': 0.001, 'taker': 0.001},
                'withdrawal_fees': {'BTC': 0.0005, 'ETH': 0.005, 'USDT': 1.0},
                'min_volumes': {'BTC': 0.001, 'ETH': 0.01, 'USDT': 10},
                'rate_limit': 1200,  # requests per minute
                'latency_estimate': 0.1  # seconds
            },
            'coinbase': {
                'fees': {'maker': 0.005, 'taker': 0.005},
                'withdrawal_fees': {'BTC': 0.0005, 'ETH': 0.0025, 'USDT': 2.5},
                'min_volumes': {'BTC': 0.001, 'ETH': 0.01, 'USDT': 1},
                'rate_limit': 600,
                'latency_estimate': 0.2
            },
            'kraken': {
                'fees': {'maker': 0.0016, 'taker': 0.0026},
                'withdrawal_fees': {'BTC': 0.00015, 'ETH': 0.0025, 'USDT': 5.0},
                'min_volumes': {'BTC': 0.0001, 'ETH': 0.001, 'USDT': 5},
                'rate_limit': 120,
                'latency_estimate': 0.3
            },
            'okx': {
                'fees': {'maker': 0.001, 'taker': 0.0015},
                'withdrawal_fees': {'BTC': 0.0004, 'ETH': 0.005, 'USDT': 1.0},
                'min_volumes': {'BTC': 0.00001, 'ETH': 0.0001, 'USDT': 1},
                'rate_limit': 2400,
                'latency_estimate': 0.15
            }
        }
    
    async def initialize_exchanges(self):
        """Initialize exchange connections."""
        crypto_logger.logger.info("Initializing exchange connections for arbitrage detection")
        
        try:
            # Initialize Binance (most liquid)
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY if hasattr(self.config, 'BINANCE_API_KEY') else None,
                'secret': self.config.BINANCE_SECRET_KEY if hasattr(self.config, 'BINANCE_SECRET_KEY') else None,
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Initialize other exchanges (public API only for price data)
            self.exchanges['coinbase'] = ccxt.coinbasepro({
                'enableRateLimit': True,
            })
            
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True,
            })
            
            self.exchanges['okx'] = ccxt.okx({
                'enableRateLimit': True,
            })
            
            # Load markets for each exchange
            for name, exchange in self.exchanges.items():
                try:
                    await exchange.load_markets()
                    crypto_logger.logger.info(f"✓ {name.capitalize()} connected - {len(exchange.markets)} markets")
                except Exception as e:
                    crypto_logger.logger.warning(f"Failed to connect to {name}: {e}")
                    del self.exchanges[name]
            
            crypto_logger.logger.info(f"Successfully initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing exchanges: {e}")
    
    async def fetch_all_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch prices from all exchanges simultaneously."""
        all_prices = {}
        
        async def fetch_exchange_prices(exchange_name: str, exchange: ccxt.Exchange):
            exchange_prices = {}
            
            for symbol in symbols:
                try:
                    # Normalize symbol for each exchange
                    normalized_symbol = self._normalize_symbol(symbol, exchange_name)
                    
                    if normalized_symbol in exchange.markets:
                        ticker = await exchange.fetch_ticker(normalized_symbol)
                        
                        exchange_prices[symbol] = {
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'last': ticker['last'],
                            'volume': ticker['baseVolume'],
                            'timestamp': ticker['timestamp'],
                            'spread': (ticker['ask'] - ticker['bid']) / ticker['last'] * 100 if ticker['bid'] and ticker['ask'] else 0
                        }
                        
                except Exception as e:
                    crypto_logger.logger.debug(f"Error fetching {symbol} from {exchange_name}: {e}")
            
            return exchange_name, exchange_prices
        
        # Fetch from all exchanges concurrently
        tasks = [
            fetch_exchange_prices(name, exchange) 
            for name, exchange in self.exchanges.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                exchange_name, prices = result
                all_prices[exchange_name] = prices
            elif isinstance(result, Exception):
                crypto_logger.logger.error(f"Exchange fetch error: {result}")
        
        # Update cache
        self.price_cache.update(all_prices)
        
        return all_prices
    
    async def fetch_orderbooks(self, symbols: List[str], depth: int = 10) -> Dict[str, Dict[str, Any]]:
        """Fetch order books for better arbitrage analysis."""
        orderbooks = {}
        
        async def fetch_exchange_orderbook(exchange_name: str, exchange: ccxt.Exchange):
            exchange_books = {}
            
            for symbol in symbols:
                try:
                    normalized_symbol = self._normalize_symbol(symbol, exchange_name)
                    
                    if normalized_symbol in exchange.markets:
                        orderbook = await exchange.fetch_order_book(normalized_symbol, depth)
                        
                        # Calculate effective prices for different trade sizes
                        effective_prices = self._calculate_effective_prices(orderbook)
                        
                        exchange_books[symbol] = {
                            'bids': orderbook['bids'][:depth],
                            'asks': orderbook['asks'][:depth],
                            'timestamp': orderbook['timestamp'],
                            'effective_prices': effective_prices
                        }
                        
                except Exception as e:
                    crypto_logger.logger.debug(f"Error fetching orderbook {symbol} from {exchange_name}: {e}")
            
            return exchange_name, exchange_books
        
        tasks = [
            fetch_exchange_orderbook(name, exchange)
            for name, exchange in self.exchanges.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple):
                exchange_name, books = result
                orderbooks[exchange_name] = books
        
        self.orderbook_cache.update(orderbooks)
        return orderbooks
    
    def _calculate_effective_prices(self, orderbook: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effective prices for different trade sizes."""
        effective_prices = {}
        
        trade_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]  # BTC equivalent sizes
        
        for size in trade_sizes:
            # Buy price (using asks)
            total_cost = 0
            remaining_size = size
            
            for ask_price, ask_volume in orderbook['asks']:
                if remaining_size <= 0:
                    break
                
                trade_volume = min(remaining_size, ask_volume)
                total_cost += trade_volume * ask_price
                remaining_size -= trade_volume
            
            if remaining_size <= 0:
                effective_prices[f'buy_{size}'] = total_cost / size
            
            # Sell price (using bids)
            total_received = 0
            remaining_size = size
            
            for bid_price, bid_volume in orderbook['bids']:
                if remaining_size <= 0:
                    break
                
                trade_volume = min(remaining_size, bid_volume)
                total_received += trade_volume * bid_price
                remaining_size -= trade_volume
            
            if remaining_size <= 0:
                effective_prices[f'sell_{size}'] = total_received / size
        
        return effective_prices
    
    def detect_simple_arbitrage(self, prices: Dict[str, Dict[str, Any]]) -> List[ArbitrageOpportunity]:
        """Detect simple arbitrage opportunities between exchanges."""
        opportunities = []
        
        # Get common symbols across exchanges
        common_symbols = self._get_common_symbols(prices)
        
        for symbol in common_symbols:
            exchange_data = []
            
            # Collect price data from all exchanges
            for exchange_name, exchange_prices in prices.items():
                if symbol in exchange_prices:
                    data = exchange_prices[symbol]
                    exchange_data.append({
                        'exchange': exchange_name,
                        'bid': data['bid'],
                        'ask': data['ask'],
                        'volume': data['volume'],
                        'spread': data['spread'],
                        'timestamp': data['timestamp']
                    })
            
            if len(exchange_data) < 2:
                continue
            
            # Find arbitrage opportunities
            for i, buy_exchange in enumerate(exchange_data):
                for j, sell_exchange in enumerate(exchange_data):
                    if i >= j:
                        continue
                    
                    # Calculate potential profit
                    buy_price = buy_exchange['ask']  # Buy at ask price
                    sell_price = sell_exchange['bid']  # Sell at bid price
                    
                    if not buy_price or not sell_price:
                        continue
                    
                    spread_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    if spread_pct > self.min_spread_threshold:
                        # Calculate net profit after fees
                        net_profit = self._calculate_net_profit(
                            buy_exchange['exchange'], sell_exchange['exchange'],
                            buy_price, sell_price, 1.0, symbol
                        )
                        
                        if net_profit > 0:
                            # Calculate available volume
                            available_volume = min(
                                buy_exchange['volume'] * 0.1,  # 10% of volume
                                sell_exchange['volume'] * 0.1
                            )
                            
                            # Risk assessment
                            risk_score = self._assess_arbitrage_risk(
                                buy_exchange, sell_exchange, symbol, spread_pct
                            )
                            
                            # Execution time estimate
                            execution_time = self._estimate_execution_time(
                                buy_exchange['exchange'], sell_exchange['exchange']
                            )
                            
                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=buy_exchange['exchange'],
                                sell_exchange=sell_exchange['exchange'],
                                buy_price=buy_price,
                                sell_price=sell_price,
                                spread_pct=spread_pct,
                                volume_available=available_volume,
                                potential_profit=net_profit * available_volume,
                                execution_time_estimate=execution_time,
                                risk_score=risk_score,
                                confidence=self._calculate_arbitrage_confidence(
                                    buy_exchange, sell_exchange, spread_pct
                                ),
                                timestamp=datetime.now(),
                                market_conditions={
                                    'buy_spread': buy_exchange['spread'],
                                    'sell_spread': sell_exchange['spread'],
                                    'buy_volume': buy_exchange['volume'],
                                    'sell_volume': sell_exchange['volume']
                                }
                            )
                            
                            opportunities.append(opportunity)
        
        # Sort by potential profit
        opportunities.sort(key=lambda x: x.potential_profit, reverse=True)
        
        # Filter by risk tolerance
        filtered_opportunities = [
            opp for opp in opportunities 
            if opp.risk_score <= self.risk_tolerance and 
               opp.execution_time_estimate <= self.max_execution_time
        ]
        
        return filtered_opportunities[:10]  # Top 10 opportunities
    
    def detect_triangular_arbitrage(self, prices: Dict[str, Dict[str, Any]]) -> List[TriangularArbitrage]:
        """Detect triangular arbitrage opportunities within exchanges."""
        opportunities = []
        
        # Common triangular pairs
        triangular_pairs = [
            ('BTC', 'ETH', 'USDT'),
            ('BTC', 'ETH', 'BUSD'),
            ('ETH', 'ADA', 'USDT'),
            ('BTC', 'SOL', 'USDT'),
            ('ETH', 'DOT', 'USDT'),
            ('BTC', 'LINK', 'USDT')
        ]
        
        for exchange_name, exchange_prices in prices.items():
            for asset_a, asset_b, base in triangular_pairs:
                try:
                    # Required pairs for triangular arbitrage
                    pair_ab = f"{asset_a}/{asset_b}"
                    pair_a_base = f"{asset_a}/{base}"
                    pair_b_base = f"{asset_b}/{base}"
                    
                    # Check if all pairs are available
                    if not all(pair in exchange_prices for pair in [pair_ab, pair_a_base, pair_b_base]):
                        continue
                    
                    # Get prices
                    price_ab = exchange_prices[pair_ab]['last']
                    price_a_base = exchange_prices[pair_a_base]['last']
                    price_b_base = exchange_prices[pair_b_base]['last']
                    
                    if not all([price_ab, price_a_base, price_b_base]):
                        continue
                    
                    # Calculate triangular arbitrage
                    # Path 1: USDT -> A -> B -> USDT
                    path1_result = (1 / price_a_base) * price_ab * price_b_base
                    path1_profit = (path1_result - 1) * 100
                    
                    # Path 2: USDT -> B -> A -> USDT  
                    path2_result = (1 / price_b_base) * (1 / price_ab) * price_a_base
                    path2_profit = (path2_result - 1) * 100
                    
                    # Check if profitable after fees
                    exchange_config = self.exchange_configs.get(exchange_name, {})
                    trading_fee = exchange_config.get('fees', {}).get('taker', 0.001)
                    total_fees = trading_fee * 3  # 3 trades
                    
                    min_profit_threshold = total_fees * 100 + 0.1  # Fees + 0.1% minimum
                    
                    if path1_profit > min_profit_threshold:
                        # Calculate volume constraints
                        volume_constraint = self._calculate_triangular_volume_constraint(
                            exchange_name, [pair_a_base, pair_ab, pair_b_base], exchange_prices
                        )
                        
                        opportunities.append(TriangularArbitrage(
                            symbol_a=asset_a,
                            symbol_b=asset_b,
                            base_currency=base,
                            exchange=exchange_name,
                            buy_a_with_base=price_a_base,
                            sell_a_for_b=price_ab,
                            sell_b_for_base=price_b_base,
                            profit_pct=path1_profit - total_fees * 100,
                            execution_path=[f"{base}->{asset_a}", f"{asset_a}->{asset_b}", f"{asset_b}->{base}"],
                            volume_constraint=volume_constraint,
                            timestamp=datetime.now()
                        ))
                    
                    if path2_profit > min_profit_threshold:
                        volume_constraint = self._calculate_triangular_volume_constraint(
                            exchange_name, [pair_b_base, pair_ab, pair_a_base], exchange_prices
                        )
                        
                        opportunities.append(TriangularArbitrage(
                            symbol_a=asset_b,
                            symbol_b=asset_a,
                            base_currency=base,
                            exchange=exchange_name,
                            buy_a_with_base=price_b_base,
                            sell_a_for_b=1/price_ab,
                            sell_b_for_base=price_a_base,
                            profit_pct=path2_profit - total_fees * 100,
                            execution_path=[f"{base}->{asset_b}", f"{asset_b}->{asset_a}", f"{asset_a}->{base}"],
                            volume_constraint=volume_constraint,
                            timestamp=datetime.now()
                        ))
                
                except Exception as e:
                    crypto_logger.logger.debug(f"Error in triangular arbitrage calculation: {e}")
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        return opportunities[:5]  # Top 5 triangular opportunities
    
    def _normalize_symbol(self, symbol: str, exchange_name: str) -> str:
        """Normalize symbol for different exchanges."""
        # Handle exchange-specific symbol formats
        symbol_map = {
            'binance': {
                'BTC/USDT': 'BTC/USDT',
                'ETH/USDT': 'ETH/USDT',
                'SOL/USDT': 'SOL/USDT'
            },
            'coinbase': {
                'BTC/USDT': 'BTC-USD',  # Coinbase uses USD instead of USDT
                'ETH/USDT': 'ETH-USD',
                'SOL/USDT': 'SOL-USD'
            },
            'kraken': {
                'BTC/USDT': 'BTC/USD',
                'ETH/USDT': 'ETH/USD',
                'SOL/USDT': 'SOL/USD'
            }
        }
        
        return symbol_map.get(exchange_name, {}).get(symbol, symbol)
    
    def _get_common_symbols(self, prices: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get symbols common across exchanges."""
        if not prices:
            return []
        
        # Start with symbols from first exchange
        common = set(next(iter(prices.values())).keys())
        
        # Intersect with symbols from other exchanges
        for exchange_prices in prices.values():
            common &= set(exchange_prices.keys())
        
        return list(common)
    
    def _calculate_net_profit(self, buy_exchange: str, sell_exchange: str, 
                            buy_price: float, sell_price: float, 
                            volume: float, symbol: str) -> float:
        """Calculate net profit after all fees."""
        # Trading fees
        buy_config = self.exchange_configs.get(buy_exchange, {})
        sell_config = self.exchange_configs.get(sell_exchange, {})
        
        buy_fee = buy_config.get('fees', {}).get('taker', 0.001)
        sell_fee = sell_config.get('fees', {}).get('taker', 0.001)
        
        # Withdrawal fees
        base_symbol = symbol.split('/')[0]
        withdrawal_fee = buy_config.get('withdrawal_fees', {}).get(base_symbol, 0)
        
        # Calculate costs
        buy_cost = buy_price * volume * (1 + buy_fee)
        sell_revenue = sell_price * volume * (1 - sell_fee)
        transfer_cost = withdrawal_fee * buy_price  # Convert to USD equivalent
        
        net_profit = sell_revenue - buy_cost - transfer_cost
        
        return net_profit / buy_cost  # Return as ratio
    
    def _assess_arbitrage_risk(self, buy_data: Dict, sell_data: Dict, 
                             symbol: str, spread_pct: float) -> float:
        """Assess risk score for arbitrage opportunity."""
        risk_factors = []
        
        # Spread size risk (larger spreads = higher risk)
        if spread_pct > 5:
            risk_factors.append(0.3)
        elif spread_pct > 2:
            risk_factors.append(0.1)
        
        # Volume risk (low volume = higher risk)
        min_volume = min(buy_data['volume'], sell_data['volume'])
        if min_volume < 1:  # Less than 1 BTC equivalent
            risk_factors.append(0.4)
        elif min_volume < 10:
            risk_factors.append(0.2)
        
        # Exchange spread risk (wide spreads = illiquid market)
        avg_spread = (buy_data['spread'] + sell_data['spread']) / 2
        if avg_spread > 0.5:
            risk_factors.append(0.3)
        elif avg_spread > 0.2:
            risk_factors.append(0.1)
        
        # Time decay risk (old prices = higher risk)
        current_time = datetime.now().timestamp() * 1000
        buy_age = (current_time - buy_data['timestamp']) / 1000  # seconds
        sell_age = (current_time - sell_data['timestamp']) / 1000
        
        if max(buy_age, sell_age) > 30:  # More than 30 seconds old
            risk_factors.append(0.4)
        elif max(buy_age, sell_age) > 10:
            risk_factors.append(0.2)
        
        # Exchange reliability (could add exchange-specific risk scores)
        exchange_risk = {
            'binance': 0.1,
            'coinbase': 0.1,
            'kraken': 0.2,
            'okx': 0.2,
        }
        
        buy_exchange_risk = exchange_risk.get(buy_data['exchange'], 0.3)
        sell_exchange_risk = exchange_risk.get(sell_data['exchange'], 0.3)
        risk_factors.extend([buy_exchange_risk, sell_exchange_risk])
        
        # Calculate final risk score (0-1, lower is better)
        return min(sum(risk_factors), 1.0)
    
    def _estimate_execution_time(self, buy_exchange: str, sell_exchange: str) -> float:
        """Estimate total execution time for arbitrage."""
        buy_latency = self.exchange_configs.get(buy_exchange, {}).get('latency_estimate', 0.5)
        sell_latency = self.exchange_configs.get(sell_exchange, {}).get('latency_estimate', 0.5)
        
        # Add transfer time if different exchanges (assume instant for same exchange)
        transfer_time = 0 if buy_exchange == sell_exchange else 300  # 5 minutes for transfers
        
        return buy_latency + sell_latency + transfer_time
    
    def _calculate_arbitrage_confidence(self, buy_data: Dict, sell_data: Dict, spread_pct: float) -> float:
        """Calculate confidence score for arbitrage opportunity."""
        confidence_factors = []
        
        # Spread confidence (moderate spreads are more reliable)
        if 0.5 <= spread_pct <= 3:
            confidence_factors.append(0.4)
        elif spread_pct > 3:
            confidence_factors.append(0.2)  # Too good to be true
        else:
            confidence_factors.append(0.1)  # Too small
        
        # Volume confidence
        min_volume = min(buy_data['volume'], sell_data['volume'])
        if min_volume > 10:
            confidence_factors.append(0.3)
        elif min_volume > 1:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Price freshness
        current_time = datetime.now().timestamp() * 1000
        max_age = max(
            current_time - buy_data['timestamp'],
            current_time - sell_data['timestamp']
        ) / 1000
        
        if max_age < 5:
            confidence_factors.append(0.3)
        elif max_age < 15:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors) * 100, 95)  # 0-95% confidence
    
    def _calculate_triangular_volume_constraint(self, exchange: str, pairs: List[str], 
                                              prices: Dict[str, Any]) -> float:
        """Calculate volume constraint for triangular arbitrage."""
        min_volumes = []
        
        for pair in pairs:
            if pair in prices:
                volume = prices[pair].get('volume', 0)
                min_volumes.append(volume * 0.05)  # 5% of volume
        
        return min(min_volumes) if min_volumes else 0
    
    async def monitor_arbitrage_opportunities(self, symbols: List[str], 
                                           callback_func=None) -> None:
        """Continuously monitor for arbitrage opportunities."""
        crypto_logger.logger.info("Starting arbitrage monitoring")
        
        try:
            while True:
                # Fetch prices from all exchanges
                prices = await self.fetch_all_prices(symbols)
                
                if not prices:
                    await asyncio.sleep(5)
                    continue
                
                # Detect simple arbitrage
                simple_opportunities = self.detect_simple_arbitrage(prices)
                
                # Detect triangular arbitrage
                triangular_opportunities = self.detect_triangular_arbitrage(prices)
                
                # Log significant opportunities
                for opp in simple_opportunities[:3]:  # Top 3 only
                    if opp.spread_pct > 1.0:  # Only log >1% spreads
                        crypto_logger.logger.info(
                            f"Arbitrage: {opp.symbol} - Buy {opp.buy_exchange} @ ${opp.buy_price:.4f}, "
                            f"Sell {opp.sell_exchange} @ ${opp.sell_price:.4f} - "
                            f"Spread: {opp.spread_pct:.2f}% - Profit: ${opp.potential_profit:.2f}"
                        )
                
                for opp in triangular_opportunities[:2]:  # Top 2 triangular
                    if opp.profit_pct > 0.5:
                        crypto_logger.logger.info(
                            f"Triangular: {opp.symbol_a}-{opp.symbol_b}-{opp.base_currency} "
                            f"on {opp.exchange} - Profit: {opp.profit_pct:.2f}%"
                        )
                
                # Call callback if provided
                if callback_func:
                    await callback_func({
                        'simple': simple_opportunities,
                        'triangular': triangular_opportunities,
                        'timestamp': datetime.now()
                    })
                
                # Wait before next scan
                await asyncio.sleep(2)  # 2-second intervals for high-frequency monitoring
                
        except Exception as e:
            crypto_logger.logger.error(f"Error in arbitrage monitoring: {e}")
        finally:
            await self.close_connections()
    
    async def close_connections(self):
        """Close all exchange connections."""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                crypto_logger.logger.info(f"Closed connection to {name}")
            except:
                pass
    
    def get_arbitrage_summary(self) -> Dict[str, Any]:
        """Get summary of recent arbitrage activity."""
        return {
            'exchanges_monitored': list(self.exchanges.keys()),
            'last_update': datetime.now().isoformat(),
            'price_cache_size': len(self.price_cache),
            'orderbook_cache_size': len(self.orderbook_cache),
            'min_spread_threshold': self.min_spread_threshold,
            'risk_tolerance': self.risk_tolerance,
            'max_execution_time': self.max_execution_time
        }

# Global arbitrage detector instance
arbitrage_detector = MultiExchangeArbitrageDetector()