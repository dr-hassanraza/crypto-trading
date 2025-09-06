import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math
from decimal import Decimal

from src.utils.logging_config import crypto_logger
from config.config import Config

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class SmartOrder:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    order_type: OrderType
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    strategy: str  # 'smart_routing', 'iceberg', 'twap', 'vwap'
    target_exchanges: List[str]
    max_slippage: float
    max_execution_time: int  # seconds
    created_at: datetime
    status: OrderStatus
    filled_amount: float
    avg_fill_price: float
    total_fees: float
    sub_orders: List[Dict]
    execution_stats: Dict[str, Any]

@dataclass
class ExecutionVenue:
    exchange: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_pct: float
    volume_24h: float
    liquidity_score: float
    latency_ms: float
    fees: Dict[str, float]
    is_available: bool

@dataclass
class MarketImpactModel:
    symbol: str
    exchange: str
    linear_coefficient: float
    sqrt_coefficient: float
    log_coefficient: float
    min_impact: float
    max_impact: float
    volume_threshold: float

class SmartOrderRouter:
    """Advanced order execution engine with smart routing and algorithms."""
    
    def __init__(self):
        self.config = Config()
        self.exchanges = {}
        self.active_orders = {}
        self.execution_history = []
        self.market_impact_models = {}
        self.venue_latencies = {}
        
        # Execution parameters
        self.max_order_size_pct = 0.1  # Max 10% of daily volume
        self.min_fill_threshold = 0.95  # Minimum 95% fill required
        self.max_spread_tolerance = 0.005  # Max 0.5% spread tolerance
        
        # Exchange configurations
        self.exchange_configs = {
            'binance': {
                'fees': {'maker': 0.001, 'taker': 0.001},
                'min_notional': {'BTCUSDT': 10, 'ETHUSDT': 10},
                'max_order_size': 1000000,
                'supports_iceberg': True,
                'supports_stop_loss': True,
                'api_rate_limit': 1200,
                'typical_latency': 50  # ms
            },
            'coinbase': {
                'fees': {'maker': 0.005, 'taker': 0.005},
                'min_notional': {'BTCUSDT': 1, 'ETHUSDT': 1},
                'max_order_size': 500000,
                'supports_iceberg': False,
                'supports_stop_loss': True,
                'api_rate_limit': 300,
                'typical_latency': 100
            },
            'kraken': {
                'fees': {'maker': 0.0016, 'taker': 0.0026},
                'min_notional': {'BTCUSDT': 5, 'ETHUSDT': 5},
                'max_order_size': 200000,
                'supports_iceberg': False,
                'supports_stop_loss': True,
                'api_rate_limit': 60,
                'typical_latency': 150
            }
        }
    
    async def initialize_exchanges(self):
        """Initialize exchange connections for trading."""
        crypto_logger.logger.info("Initializing trading exchanges")
        
        try:
            # Initialize exchanges with API credentials
            if hasattr(self.config, 'BINANCE_API_KEY') and self.config.BINANCE_API_KEY:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config.BINANCE_API_KEY,
                    'secret': self.config.BINANCE_SECRET_KEY,
                    'sandbox': False,  # Set to True for testing
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            
            # Initialize paper trading exchanges if no real credentials
            if not self.exchanges:
                crypto_logger.logger.warning("No exchange credentials found - initializing paper trading mode")
                self._initialize_paper_trading()
            
            # Load markets and test connections
            for name, exchange in self.exchanges.items():
                try:
                    await exchange.load_markets()
                    balance = await exchange.fetch_balance()
                    crypto_logger.logger.info(f"✓ {name.capitalize()} trading connection established")
                except Exception as e:
                    crypto_logger.logger.error(f"Failed to connect to {name}: {e}")
                    del self.exchanges[name]
            
            # Initialize market impact models
            await self._initialize_market_impact_models()
            
            crypto_logger.logger.info(f"Trading engine initialized with {len(self.exchanges)} exchanges")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing trading exchanges: {e}")
    
    def _initialize_paper_trading(self):
        """Initialize paper trading mode for testing."""
        self.paper_trading = True
        self.paper_balance = {
            'USDT': 100000.0,  # $100k starting balance
            'BTC': 0.0,
            'ETH': 0.0
        }
        crypto_logger.logger.info("Paper trading mode activated")
    
    async def _initialize_market_impact_models(self):
        """Initialize market impact models for different symbols and exchanges."""
        
        # Mock market impact models (in production, these would be calibrated from historical data)
        impact_models = {
            ('BTCUSDT', 'binance'): MarketImpactModel(
                symbol='BTCUSDT',
                exchange='binance',
                linear_coefficient=0.0001,
                sqrt_coefficient=0.001,
                log_coefficient=0.0005,
                min_impact=0.0001,
                max_impact=0.05,
                volume_threshold=1000000
            ),
            ('ETHUSDT', 'binance'): MarketImpactModel(
                symbol='ETHUSDT',
                exchange='binance',
                linear_coefficient=0.0002,
                sqrt_coefficient=0.0015,
                log_coefficient=0.0008,
                min_impact=0.0001,
                max_impact=0.08,
                volume_threshold=500000
            )
        }
        
        self.market_impact_models = impact_models
        crypto_logger.logger.info(f"Initialized {len(impact_models)} market impact models")
    
    async def execute_smart_order(self, order_request: Dict[str, Any]) -> SmartOrder:
        """Execute a smart order with optimal routing and algorithms."""
        
        # Create smart order object
        smart_order = SmartOrder(
            id=f"smart_{int(datetime.now().timestamp() * 1000)}",
            symbol=order_request['symbol'],
            side=order_request['side'],
            amount=float(order_request['amount']),
            order_type=OrderType(order_request.get('type', 'market')),
            price=order_request.get('price'),
            stop_price=order_request.get('stopPrice'),
            time_in_force=order_request.get('timeInForce', 'GTC'),
            strategy=order_request.get('strategy', 'smart_routing'),
            target_exchanges=order_request.get('exchanges', list(self.exchanges.keys())),
            max_slippage=order_request.get('maxSlippage', 0.01),
            max_execution_time=order_request.get('maxExecutionTime', 300),
            created_at=datetime.now(),
            status=OrderStatus.PENDING,
            filled_amount=0.0,
            avg_fill_price=0.0,
            total_fees=0.0,
            sub_orders=[],
            execution_stats={}
        )
        
        crypto_logger.logger.info(
            f"🎯 Executing smart order: {smart_order.side.upper()} {smart_order.amount} "
            f"{smart_order.symbol} via {smart_order.strategy}"
        )
        
        try:
            # Execute based on strategy
            if smart_order.strategy == 'smart_routing':
                await self._execute_smart_routing(smart_order)
            elif smart_order.strategy == 'iceberg':
                await self._execute_iceberg_order(smart_order)
            elif smart_order.strategy == 'twap':
                await self._execute_twap_order(smart_order)
            elif smart_order.strategy == 'vwap':
                await self._execute_vwap_order(smart_order)
            else:
                await self._execute_simple_order(smart_order)
            
            # Store in active orders
            self.active_orders[smart_order.id] = smart_order
            
            # Log execution results
            self._log_execution_results(smart_order)
            
            return smart_order
            
        except Exception as e:
            smart_order.status = OrderStatus.REJECTED
            crypto_logger.logger.error(f"Smart order execution failed: {e}")
            return smart_order
    
    async def _execute_smart_routing(self, order: SmartOrder):
        """Execute order with smart routing across multiple exchanges."""
        
        # Get best execution venues
        venues = await self._get_execution_venues(order.symbol, order.target_exchanges)
        
        if not venues:
            raise Exception("No available execution venues")
        
        # Calculate optimal routing
        routing_plan = self._calculate_optimal_routing(order, venues)
        
        order.status = OrderStatus.OPEN
        execution_tasks = []
        
        for route in routing_plan:
            # Create sub-order for each route
            sub_order_task = self._execute_sub_order(order, route)
            execution_tasks.append(sub_order_task)
        
        # Execute all sub-orders concurrently
        sub_order_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for result in sub_order_results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                total_filled += result.get('filled', 0)
                total_cost += result.get('cost', 0)
                total_fees += result.get('fee', 0)
                order.sub_orders.append(result)
        
        # Update order status
        order.filled_amount = total_filled
        order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        order.total_fees = total_fees
        
        if total_filled >= order.amount * self.min_fill_threshold:
            order.status = OrderStatus.CLOSED
        elif total_filled > 0:
            order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.REJECTED
        
        # Calculate execution statistics
        order.execution_stats = self._calculate_execution_stats(order, venues)
    
    async def _execute_iceberg_order(self, order: SmartOrder):
        """Execute large order using iceberg strategy to minimize market impact."""
        
        # Calculate iceberg parameters
        total_amount = order.amount
        slice_size = min(total_amount * 0.1, 10000)  # 10% slices, max $10k per slice
        num_slices = math.ceil(total_amount / slice_size)
        
        crypto_logger.logger.info(f"Executing iceberg order: {num_slices} slices of {slice_size}")
        
        order.status = OrderStatus.OPEN
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for slice_num in range(num_slices):
            current_slice_size = min(slice_size, total_amount - total_filled)
            
            if current_slice_size <= 0:
                break
            
            # Execute slice
            slice_result = await self._execute_order_slice(
                order.symbol,
                order.side,
                current_slice_size,
                order.target_exchanges[0] if order.target_exchanges else 'binance'
            )
            
            if slice_result:
                total_filled += slice_result.get('filled', 0)
                total_cost += slice_result.get('cost', 0)
                total_fees += slice_result.get('fee', 0)
                order.sub_orders.append(slice_result)
                
                # Wait between slices to avoid detection
                if slice_num < num_slices - 1:
                    await asyncio.sleep(np.random.uniform(10, 30))  # 10-30 second delay
            
            # Check if we should stop (e.g., due to adverse price movement)
            if self._should_stop_iceberg(order, slice_result):
                break
        
        # Update order
        order.filled_amount = total_filled
        order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        order.total_fees = total_fees
        order.status = OrderStatus.CLOSED if total_filled >= order.amount * 0.95 else OrderStatus.PARTIAL
    
    async def _execute_twap_order(self, order: SmartOrder):
        """Execute order using Time-Weighted Average Price strategy."""
        
        # TWAP parameters
        execution_window = min(order.max_execution_time, 3600)  # Max 1 hour
        num_intervals = 12  # Execute every 5 minutes for 1 hour
        interval_duration = execution_window / num_intervals
        amount_per_interval = order.amount / num_intervals
        
        crypto_logger.logger.info(
            f"Executing TWAP order: {num_intervals} intervals of {amount_per_interval} "
            f"every {interval_duration:.0f} seconds"
        )
        
        order.status = OrderStatus.OPEN
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for interval in range(num_intervals):
            # Execute interval amount
            interval_result = await self._execute_order_slice(
                order.symbol,
                order.side,
                amount_per_interval,
                order.target_exchanges[0] if order.target_exchanges else 'binance'
            )
            
            if interval_result:
                total_filled += interval_result.get('filled', 0)
                total_cost += interval_result.get('cost', 0)
                total_fees += interval_result.get('fee', 0)
                order.sub_orders.append(interval_result)
            
            # Wait for next interval (except last one)
            if interval < num_intervals - 1:
                await asyncio.sleep(interval_duration)
        
        # Update order
        order.filled_amount = total_filled
        order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        order.total_fees = total_fees
        order.status = OrderStatus.CLOSED if total_filled >= order.amount * 0.95 else OrderStatus.PARTIAL
    
    async def _execute_vwap_order(self, order: SmartOrder):
        """Execute order using Volume-Weighted Average Price strategy."""
        
        # Get historical volume profile
        volume_profile = await self._get_volume_profile(order.symbol)
        
        if not volume_profile:
            # Fallback to TWAP if no volume data
            await self._execute_twap_order(order)
            return
        
        # Calculate execution schedule based on volume profile
        execution_schedule = self._calculate_vwap_schedule(order, volume_profile)
        
        crypto_logger.logger.info(f"Executing VWAP order with {len(execution_schedule)} scheduled executions")
        
        order.status = OrderStatus.OPEN
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for scheduled_execution in execution_schedule:
            # Wait until scheduled time
            wait_time = (scheduled_execution['time'] - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 300))  # Max 5 minute wait
            
            # Execute scheduled amount
            execution_result = await self._execute_order_slice(
                order.symbol,
                order.side,
                scheduled_execution['amount'],
                order.target_exchanges[0] if order.target_exchanges else 'binance'
            )
            
            if execution_result:
                total_filled += execution_result.get('filled', 0)
                total_cost += execution_result.get('cost', 0)
                total_fees += execution_result.get('fee', 0)
                order.sub_orders.append(execution_result)
        
        # Update order
        order.filled_amount = total_filled
        order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        order.total_fees = total_fees
        order.status = OrderStatus.CLOSED if total_filled >= order.amount * 0.95 else OrderStatus.PARTIAL
    
    async def _execute_simple_order(self, order: SmartOrder):
        """Execute simple market or limit order."""
        
        exchange_name = order.target_exchanges[0] if order.target_exchanges else 'binance'
        
        if hasattr(self, 'paper_trading') and self.paper_trading:
            # Paper trading execution
            result = await self._execute_paper_trade(order)
        else:
            # Real trading execution
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not available")
            
            try:
                if order.order_type == OrderType.MARKET:
                    result = await exchange.create_market_order(
                        order.symbol, order.side, order.amount
                    )
                elif order.order_type == OrderType.LIMIT:
                    result = await exchange.create_limit_order(
                        order.symbol, order.side, order.amount, order.price
                    )
                else:
                    raise Exception(f"Order type {order.order_type} not supported")
                    
            except Exception as e:
                crypto_logger.logger.error(f"Order execution failed on {exchange_name}: {e}")
                result = None
        
        if result:
            order.filled_amount = result.get('filled', 0)
            order.avg_fill_price = result.get('average', result.get('price', 0))
            order.total_fees = result.get('fee', {}).get('cost', 0)
            order.status = OrderStatus.CLOSED if order.filled_amount >= order.amount * 0.95 else OrderStatus.PARTIAL
            order.sub_orders = [result]
        else:
            order.status = OrderStatus.REJECTED
    
    async def _get_execution_venues(self, symbol: str, target_exchanges: List[str]) -> List[ExecutionVenue]:
        """Get available execution venues with current market data."""
        
        venues = []
        
        for exchange_name in target_exchanges:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                
                # Get order book
                orderbook = await exchange.fetch_order_book(symbol, limit=10)
                
                # Get 24h volume
                ticker = await exchange.fetch_ticker(symbol)
                
                # Calculate venue metrics
                bid_price = orderbook['bids'][0][0] if orderbook['bids'] else 0
                ask_price = orderbook['asks'][0][0] if orderbook['asks'] else 0
                bid_size = orderbook['bids'][0][1] if orderbook['bids'] else 0
                ask_size = orderbook['asks'][0][1] if orderbook['asks'] else 0
                
                spread_pct = ((ask_price - bid_price) / ask_price) * 100 if ask_price > 0 else 0
                liquidity_score = self._calculate_liquidity_score(orderbook, ticker['quoteVolume'])
                
                venue = ExecutionVenue(
                    exchange=exchange_name,
                    symbol=symbol,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    spread_pct=spread_pct,
                    volume_24h=ticker['quoteVolume'],
                    liquidity_score=liquidity_score,
                    latency_ms=self.exchange_configs[exchange_name]['typical_latency'],
                    fees=self.exchange_configs[exchange_name]['fees'],
                    is_available=True
                )
                
                venues.append(venue)
                
            except Exception as e:
                crypto_logger.logger.debug(f"Error getting venue data for {exchange_name}: {e}")
        
        return venues
    
    def _calculate_optimal_routing(self, order: SmartOrder, venues: List[ExecutionVenue]) -> List[Dict[str, Any]]:
        """Calculate optimal order routing across venues."""
        
        # Sort venues by execution quality score
        scored_venues = []
        
        for venue in venues:
            # Calculate execution quality score
            price_score = venue.ask_price if order.side == 'buy' else venue.bid_price
            liquidity_score = venue.liquidity_score
            cost_score = 1 / (1 + venue.fees['taker'])  # Lower fees = higher score
            latency_score = 1 / (1 + venue.latency_ms / 100)  # Lower latency = higher score
            
            # Weighted combination
            quality_score = (
                price_score * 0.4 + 
                liquidity_score * 0.3 + 
                cost_score * 0.2 + 
                latency_score * 0.1
            )
            
            scored_venues.append((venue, quality_score))
        
        # Sort by quality score
        scored_venues.sort(key=lambda x: x[1], reverse=True)
        
        # Create routing plan
        routing_plan = []
        remaining_amount = order.amount
        
        for venue, score in scored_venues:
            if remaining_amount <= 0:
                break
            
            # Determine allocation for this venue
            max_allocation = min(
                remaining_amount,
                venue.ask_size if order.side == 'buy' else venue.bid_size,
                order.amount * 0.5  # Max 50% to any single venue
            )
            
            if max_allocation > 0:
                routing_plan.append({
                    'venue': venue,
                    'amount': max_allocation,
                    'expected_price': venue.ask_price if order.side == 'buy' else venue.bid_price,
                    'quality_score': score
                })
                
                remaining_amount -= max_allocation
        
        return routing_plan
    
    async def _execute_sub_order(self, parent_order: SmartOrder, route: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sub-order on a specific venue."""
        
        venue = route['venue']
        amount = route['amount']
        
        try:
            if hasattr(self, 'paper_trading') and self.paper_trading:
                # Paper trading
                result = {
                    'id': f"paper_{int(datetime.now().timestamp() * 1000)}",
                    'symbol': parent_order.symbol,
                    'side': parent_order.side,
                    'amount': amount,
                    'filled': amount,
                    'cost': amount * route['expected_price'],
                    'average': route['expected_price'],
                    'fee': {'cost': amount * route['expected_price'] * venue.fees['taker']},
                    'timestamp': datetime.now().timestamp() * 1000,
                    'status': 'closed',
                    'venue': venue.exchange
                }
            else:
                # Real trading
                exchange = self.exchanges[venue.exchange]
                
                if parent_order.order_type == OrderType.MARKET:
                    result = await exchange.create_market_order(
                        parent_order.symbol, parent_order.side, amount
                    )
                else:
                    price = route['expected_price']
                    result = await exchange.create_limit_order(
                        parent_order.symbol, parent_order.side, amount, price
                    )
                
                result['venue'] = venue.exchange
            
            crypto_logger.logger.info(
                f"✓ Sub-order executed: {amount} on {venue.exchange} @ {result.get('average', 0)}"
            )
            
            return result
            
        except Exception as e:
            crypto_logger.logger.error(f"Sub-order execution failed on {venue.exchange}: {e}")
            return {}
    
    async def _execute_order_slice(self, symbol: str, side: str, amount: float, exchange_name: str) -> Optional[Dict]:
        """Execute a single order slice."""
        
        try:
            if hasattr(self, 'paper_trading') and self.paper_trading:
                # Paper trading execution
                price = 45000 if 'BTC' in symbol else 2500  # Mock prices
                
                result = {
                    'id': f"paper_{int(datetime.now().timestamp() * 1000)}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'filled': amount,
                    'cost': amount * price,
                    'average': price,
                    'fee': {'cost': amount * price * 0.001},
                    'timestamp': datetime.now().timestamp() * 1000,
                    'status': 'closed'
                }
                
                # Update paper balance
                if side == 'buy':
                    self.paper_balance['USDT'] -= result['cost']
                    base_asset = symbol.replace('USDT', '')
                    self.paper_balance[base_asset] = self.paper_balance.get(base_asset, 0) + amount
                else:
                    base_asset = symbol.replace('USDT', '')
                    self.paper_balance[base_asset] -= amount
                    self.paper_balance['USDT'] += result['cost']
                
                return result
            else:
                # Real trading would go here
                exchange = self.exchanges.get(exchange_name)
                if exchange:
                    return await exchange.create_market_order(symbol, side, amount)
        
        except Exception as e:
            crypto_logger.logger.error(f"Order slice execution failed: {e}")
            return None
    
    async def _execute_paper_trade(self, order: SmartOrder) -> Dict[str, Any]:
        """Execute order in paper trading mode."""
        
        # Mock execution with realistic parameters
        base_price = 45000 if 'BTC' in order.symbol else 2500
        
        # Add some slippage for market orders
        if order.order_type == OrderType.MARKET:
            slippage = np.random.uniform(0.0005, 0.002)  # 0.05-0.2% slippage
            price = base_price * (1 + slippage) if order.side == 'buy' else base_price * (1 - slippage)
        else:
            price = order.price or base_price
        
        # Calculate fees
        fee_rate = 0.001  # 0.1% fee
        fee_cost = order.amount * price * fee_rate
        
        result = {
            'id': f"paper_{order.id}",
            'symbol': order.symbol,
            'side': order.side,
            'amount': order.amount,
            'filled': order.amount,
            'cost': order.amount * price,
            'average': price,
            'fee': {'cost': fee_cost},
            'timestamp': datetime.now().timestamp() * 1000,
            'status': 'closed'
        }
        
        return result
    
    def _calculate_liquidity_score(self, orderbook: Dict, volume_24h: float) -> float:
        """Calculate liquidity score for a venue."""
        
        # Calculate depth within 1% of mid price
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return 0
        
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        depth_1pct = 0
        
        # Sum bid depth within 1%
        for bid_price, bid_size in orderbook['bids']:
            if bid_price >= mid_price * 0.99:
                depth_1pct += bid_size * bid_price
        
        # Sum ask depth within 1%
        for ask_price, ask_size in orderbook['asks']:
            if ask_price <= mid_price * 1.01:
                depth_1pct += ask_size * ask_price
        
        # Normalize by 24h volume
        liquidity_ratio = depth_1pct / max(volume_24h * 0.01, 1)  # 1% of daily volume
        
        return min(liquidity_ratio, 1.0)  # Cap at 1.0
    
    def _should_stop_iceberg(self, order: SmartOrder, slice_result: Optional[Dict]) -> bool:
        """Determine if iceberg order should be stopped early."""
        
        if not slice_result:
            return True  # Stop if slice failed
        
        # Check for adverse price movement
        current_price = slice_result.get('average', 0)
        expected_price = order.price or current_price
        
        price_deviation = abs(current_price - expected_price) / expected_price
        
        if price_deviation > order.max_slippage:
            crypto_logger.logger.warning(f"Stopping iceberg due to excessive slippage: {price_deviation:.2%}")
            return True
        
        return False
    
    async def _get_volume_profile(self, symbol: str) -> Optional[List[Dict]]:
        """Get historical volume profile for VWAP calculation."""
        
        # Mock volume profile (in production, would fetch real historical data)
        hours = 24
        volume_profile = []
        
        for hour in range(hours):
            # Higher volume during typical trading hours
            if 8 <= hour <= 22:  # 8 AM to 10 PM UTC
                base_volume = np.random.uniform(0.8, 1.2)
            else:
                base_volume = np.random.uniform(0.3, 0.7)
            
            volume_profile.append({
                'hour': hour,
                'volume_ratio': base_volume,
                'timestamp': datetime.now() + timedelta(hours=hour-12)  # Past 12h to future 12h
            })
        
        return volume_profile
    
    def _calculate_vwap_schedule(self, order: SmartOrder, volume_profile: List[Dict]) -> List[Dict]:
        """Calculate VWAP execution schedule."""
        
        schedule = []
        total_volume_ratio = sum(vp['volume_ratio'] for vp in volume_profile)
        
        for volume_point in volume_profile:
            # Skip past time points
            if volume_point['timestamp'] < datetime.now():
                continue
            
            # Calculate amount for this time period
            volume_weight = volume_point['volume_ratio'] / total_volume_ratio
            scheduled_amount = order.amount * volume_weight
            
            if scheduled_amount > 0:
                schedule.append({
                    'time': volume_point['timestamp'],
                    'amount': scheduled_amount,
                    'volume_weight': volume_weight
                })
        
        return schedule
    
    def _calculate_execution_stats(self, order: SmartOrder, venues: List[ExecutionVenue]) -> Dict[str, Any]:
        """Calculate execution statistics for the order."""
        
        if not order.sub_orders:
            return {}
        
        # Calculate VWAP vs market price
        total_cost = sum(sub['cost'] for sub in order.sub_orders if sub.get('cost'))
        total_filled = sum(sub['filled'] for sub in order.sub_orders if sub.get('filled'))
        
        execution_vwap = total_cost / total_filled if total_filled > 0 else 0
        
        # Calculate market price (best available price across venues)
        if order.side == 'buy':
            market_price = min(venue.ask_price for venue in venues if venue.ask_price > 0)
        else:
            market_price = max(venue.bid_price for venue in venues if venue.bid_price > 0)
        
        # Calculate slippage
        slippage = ((execution_vwap - market_price) / market_price) * 100 if market_price > 0 else 0
        
        return {
            'execution_vwap': execution_vwap,
            'market_price': market_price,
            'slippage_bps': slippage * 100,  # Basis points
            'fill_rate': order.filled_amount / order.amount,
            'venues_used': len(set(sub.get('venue', 'unknown') for sub in order.sub_orders)),
            'execution_time_seconds': (datetime.now() - order.created_at).total_seconds(),
            'total_fees_bps': (order.total_fees / total_cost) * 10000 if total_cost > 0 else 0
        }
    
    def _log_execution_results(self, order: SmartOrder):
        """Log execution results for analysis."""
        
        crypto_logger.logger.info(
            f"📊 Order execution complete: {order.id} - "
            f"Filled: {order.filled_amount}/{order.amount} "
            f"({order.filled_amount/order.amount:.1%}) @ ${order.avg_fill_price:.2f}"
        )
        
        if order.execution_stats:
            stats = order.execution_stats
            crypto_logger.logger.info(
                f"   Slippage: {stats.get('slippage_bps', 0):.1f}bps, "
                f"Fees: {stats.get('total_fees_bps', 0):.1f}bps, "
                f"Venues: {stats.get('venues_used', 0)}, "
                f"Time: {stats.get('execution_time_seconds', 0):.1f}s"
            )
        
        # Store in execution history
        self.execution_history.append({
            'timestamp': order.created_at,
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'amount': order.amount,
            'filled': order.filled_amount,
            'avg_price': order.avg_fill_price,
            'total_fees': order.total_fees,
            'strategy': order.strategy,
            'status': order.status.value,
            'execution_stats': order.execution_stats
        })
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        try:
            # Cancel all sub-orders
            for sub_order in order.sub_orders:
                if sub_order.get('status') == 'open':
                    venue_name = sub_order.get('venue', order.target_exchanges[0])
                    
                    if hasattr(self, 'paper_trading'):
                        # Paper trading - just mark as canceled
                        sub_order['status'] = 'canceled'
                    else:
                        # Real trading
                        exchange = self.exchanges.get(venue_name)
                        if exchange and sub_order.get('id'):
                            await exchange.cancel_order(sub_order['id'], order.symbol)
            
            order.status = OrderStatus.CANCELED
            crypto_logger.logger.info(f"Order {order_id} canceled successfully")
            return True
            
        except Exception as e:
            crypto_logger.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_execution_report(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Generate execution performance report."""
        
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        recent_executions = [
            ex for ex in self.execution_history 
            if ex['timestamp'] >= cutoff_time
        ]
        
        if not recent_executions:
            return {}
        
        # Calculate aggregate statistics
        total_orders = len(recent_executions)
        successful_orders = len([ex for ex in recent_executions if ex['status'] == 'closed'])
        total_volume = sum(ex['amount'] * ex['avg_price'] for ex in recent_executions if ex['avg_price'])
        total_fees = sum(ex['total_fees'] for ex in recent_executions)
        
        # Calculate average metrics
        avg_slippage = np.mean([
            ex['execution_stats'].get('slippage_bps', 0) 
            for ex in recent_executions if ex.get('execution_stats')
        ])
        
        avg_execution_time = np.mean([
            ex['execution_stats'].get('execution_time_seconds', 0) 
            for ex in recent_executions if ex.get('execution_stats')
        ])
        
        return {
            'period_hours': time_period_hours,
            'summary': {
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'success_rate': successful_orders / total_orders if total_orders > 0 else 0,
                'total_volume_usd': total_volume,
                'total_fees_usd': total_fees,
                'avg_slippage_bps': avg_slippage,
                'avg_execution_time_seconds': avg_execution_time
            },
            'by_strategy': {
                strategy: {
                    'orders': len([ex for ex in recent_executions if ex['strategy'] == strategy]),
                    'avg_slippage_bps': np.mean([
                        ex['execution_stats'].get('slippage_bps', 0) 
                        for ex in recent_executions 
                        if ex['strategy'] == strategy and ex.get('execution_stats')
                    ])
                } for strategy in set(ex['strategy'] for ex in recent_executions)
            },
            'recent_orders': recent_executions[-10:],  # Last 10 orders
            'timestamp': datetime.now().isoformat()
        }

    async def get_order_status(self, order_id: str) -> Optional[SmartOrder]:
        """Get current status of an order."""
        return self.active_orders.get(order_id)
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status for paper trading."""
        if hasattr(self, 'paper_trading') and self.paper_trading:
            total_value = self.paper_balance['USDT']
            
            # Add value of crypto holdings (using mock prices)
            for asset, amount in self.paper_balance.items():
                if asset != 'USDT' and amount > 0:
                    if asset == 'BTC':
                        total_value += amount * 45000  # Mock BTC price
                    elif asset == 'ETH':
                        total_value += amount * 2500   # Mock ETH price
            
            return {
                'balances': self.paper_balance,
                'total_value_usd': total_value,
                'active_orders': len(self.active_orders),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Real trading portfolio status would require exchange API calls
            return {'message': 'Live trading portfolio status requires exchange connections'}
    
    async def cleanup(self):
        """Clean up resources and close exchange connections."""
        crypto_logger.logger.info("Cleaning up trading execution engine")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id)
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                await exchange.close()
        
        crypto_logger.logger.info("Trading execution engine cleanup complete")

# Global execution engine instance
execution_engine = SmartOrderRouter()