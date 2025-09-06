import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import deque
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    bid_count: int
    ask_count: int
    total_bid_volume: float
    total_ask_volume: float
    spread_absolute: float
    spread_percentage: float
    mid_price: float
    weighted_mid_price: float
    book_imbalance: float

@dataclass
class TradeData:
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: str
    is_aggressive: bool
    volume_weighted_price: float

@dataclass
class MarketImpactMeasure:
    symbol: str
    trade_size: float
    pre_trade_mid: float
    post_trade_mid: float
    temporary_impact: float
    permanent_impact: float
    price_recovery_time: float
    liquidity_cost: float
    spread_cost: float

@dataclass
class LiquidityMetrics:
    symbol: str
    timestamp: datetime
    effective_spread: float
    realized_spread: float
    price_impact: float
    order_flow_imbalance: float
    market_depth: float
    resilience_score: float
    tightness_score: float
    immediacy_score: float

class MarketMicrostructureAnalyzer:
    """Advanced market microstructure analysis for cryptocurrency markets."""
    
    def __init__(self):
        self.config = Config()
        self.exchanges = {}
        self.orderbook_history = {}
        self.trade_history = {}
        self.liquidity_metrics = {}
        
        # Analysis parameters
        self.orderbook_depth = 20  # Number of levels to analyze
        self.tick_analysis_window = 1000  # Number of recent trades to analyze
        self.impact_measurement_window = 30  # Seconds to measure impact
        
        # Market structure models
        self.structural_models = {}
        
        # Real-time data buffers
        self.live_orderbooks = {}
        self.live_trades = deque(maxlen=10000)
        
    async def initialize_microstructure_analysis(self):
        """Initialize market microstructure analysis system."""
        crypto_logger.logger.info("Initializing market microstructure analyzer")
        
        try:
            # Generate mock market data for analysis
            await self._generate_mock_microstructure_data()
            
            # Initialize liquidity analysis models
            await self._initialize_liquidity_models()
            
            # Setup market impact models
            await self._initialize_impact_models()
            
            crypto_logger.logger.info("✓ Market microstructure analyzer initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing microstructure analysis: {e}")
    
    async def _generate_mock_microstructure_data(self):
        """Generate realistic mock orderbook and trade data."""
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            # Generate mock orderbook snapshots
            orderbooks = []
            trades = []
            
            base_price = 45000 if 'BTC' in symbol else (2500 if 'ETH' in symbol else 100)
            
            # Generate 1000 orderbook snapshots (simulate 1 minute of data at 60ms intervals)
            for i in range(1000):
                timestamp = datetime.now() - timedelta(seconds=(1000-i) * 0.06)
                
                # Add some price drift and volatility
                price_drift = np.random.normal(0, base_price * 0.0001)
                current_mid = base_price + np.cumsum([price_drift])[0]
                
                # Generate realistic orderbook
                orderbook = self._generate_realistic_orderbook(symbol, current_mid, timestamp)
                orderbooks.append(orderbook)
                
                # Generate corresponding trades
                if i % 5 == 0:  # Trade every 5 snapshots on average
                    trade = self._generate_realistic_trade(symbol, current_mid, timestamp)
                    trades.append(trade)
            
            self.orderbook_history[symbol] = orderbooks
            self.trade_history[symbol] = trades
            
        crypto_logger.logger.info(f"Generated mock microstructure data for {len(symbols)} symbols")
    
    def _generate_realistic_orderbook(self, symbol: str, mid_price: float, timestamp: datetime) -> OrderBookSnapshot:
        """Generate a realistic orderbook snapshot."""
        
        # Simulate realistic spread (crypto markets typically 0.01-0.1%)
        spread_bps = np.random.uniform(1, 20)  # 1-20 basis points
        half_spread = mid_price * (spread_bps / 20000)  # Half spread
        
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Generate multiple price levels with realistic size distribution
        bids = []
        asks = []
        
        # Exponentially decreasing size away from mid
        for level in range(self.orderbook_depth):
            # Bid side
            bid_price = best_bid - (level * half_spread * 0.1)
            bid_size = np.random.exponential(100) * (0.9 ** level)  # Decreasing size
            bids.append((bid_price, bid_size))
            
            # Ask side
            ask_price = best_ask + (level * half_spread * 0.1)
            ask_size = np.random.exponential(100) * (0.9 ** level)  # Decreasing size
            asks.append((ask_price, ask_size))
        
        # Calculate metrics
        total_bid_volume = sum(size for _, size in bids)
        total_ask_volume = sum(size for _, size in asks)
        
        # Volume-weighted mid price
        total_volume = total_bid_volume + total_ask_volume
        if total_volume > 0:
            bid_weight = total_bid_volume / total_volume
            ask_weight = total_ask_volume / total_volume
            weighted_mid = (best_bid * bid_weight + best_ask * ask_weight)
        else:
            weighted_mid = mid_price
        
        # Order book imbalance
        book_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if total_volume > 0 else 0
        
        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=bids,
            asks=asks,
            bid_count=len(bids),
            ask_count=len(asks),
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            spread_absolute=best_ask - best_bid,
            spread_percentage=((best_ask - best_bid) / mid_price) * 100,
            mid_price=mid_price,
            weighted_mid_price=weighted_mid,
            book_imbalance=book_imbalance
        )
    
    def _generate_realistic_trade(self, symbol: str, mid_price: float, timestamp: datetime) -> TradeData:
        """Generate a realistic trade."""
        
        # Random trade characteristics
        is_buy = np.random.choice([True, False])
        side = 'buy' if is_buy else 'sell'
        
        # Trade size follows power law distribution (many small, few large)
        size = np.random.pareto(0.5) * 10 + 1  # Power law with minimum size
        
        # Trade price with some spread crossing
        spread_cross = np.random.uniform(-0.001, 0.001)  # Small price improvement/premium
        price = mid_price * (1 + spread_cross)
        
        # Aggressive vs passive (most trades are aggressive)
        is_aggressive = np.random.choice([True, False], p=[0.8, 0.2])
        
        return TradeData(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            trade_id=f"trade_{int(timestamp.timestamp() * 1000)}",
            is_aggressive=is_aggressive,
            volume_weighted_price=price  # Simplified for single trade
        )
    
    async def _initialize_liquidity_models(self):
        """Initialize liquidity measurement models."""
        
        self.liquidity_models = {
            'kyle_lambda': {},  # Kyle's lambda (market impact coefficient)
            'amihud_illiquidity': {},  # Amihud illiquidity ratio
            'roll_spread': {},  # Roll's spread estimator
            'corwin_schultz': {}  # Corwin-Schultz spread estimator
        }
        
        crypto_logger.logger.info("Initialized liquidity measurement models")
    
    async def _initialize_impact_models(self):
        """Initialize market impact models."""
        
        self.impact_models = {
            'linear_impact': {'slope': 0.001, 'intercept': 0.0},
            'sqrt_impact': {'coefficient': 0.01},
            'logarithmic_impact': {'coefficient': 0.005}
        }
        
        crypto_logger.logger.info("Initialized market impact models")
    
    async def analyze_order_flow(self, symbol: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze order flow patterns and imbalances."""
        
        if symbol not in self.trade_history:
            return {'error': f'No trade data available for {symbol}'}
        
        trades = self.trade_history[symbol]
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_trades = [t for t in trades if t.timestamp >= cutoff_time]
        
        if not recent_trades:
            return {'error': 'No recent trades found'}
        
        # Calculate order flow metrics
        buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
        sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
        total_volume = buy_volume + sell_volume
        
        # Order flow imbalance
        flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Volume-weighted average prices
        buy_vwap = sum(t.price * t.size for t in recent_trades if t.side == 'buy') / buy_volume if buy_volume > 0 else 0
        sell_vwap = sum(t.price * t.size for t in recent_trades if t.side == 'sell') / sell_volume if sell_volume > 0 else 0
        
        # Trade size analysis
        trade_sizes = [t.size for t in recent_trades]
        avg_trade_size = np.mean(trade_sizes)
        median_trade_size = np.median(trade_sizes)
        large_trade_threshold = np.percentile(trade_sizes, 90)
        
        # Identify large trades (block trades)
        large_trades = [t for t in recent_trades if t.size >= large_trade_threshold]
        
        # Trade intensity analysis
        trade_intervals = []
        for i in range(1, len(recent_trades)):
            interval = (recent_trades[i].timestamp - recent_trades[i-1].timestamp).total_seconds()
            trade_intervals.append(interval)
        
        avg_trade_interval = np.mean(trade_intervals) if trade_intervals else 0
        trade_intensity = 1 / avg_trade_interval if avg_trade_interval > 0 else 0
        
        # Price impact of trades
        price_impacts = []
        for i, trade in enumerate(recent_trades[:-1]):
            next_trade = recent_trades[i + 1]
            price_change = (next_trade.price - trade.price) / trade.price
            
            # Directional impact
            if trade.side == 'buy':
                impact = price_change
            else:
                impact = -price_change
                
            price_impacts.append(impact)
        
        avg_price_impact = np.mean(price_impacts) if price_impacts else 0
        
        return {
            'symbol': symbol,
            'time_window_minutes': time_window_minutes,
            'volume_analysis': {
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_volume_pct': (buy_volume / total_volume) * 100 if total_volume > 0 else 0,
                'sell_volume_pct': (sell_volume / total_volume) * 100 if total_volume > 0 else 0,
                'order_flow_imbalance': flow_imbalance
            },
            'price_analysis': {
                'buy_vwap': buy_vwap,
                'sell_vwap': sell_vwap,
                'vwap_spread': abs(buy_vwap - sell_vwap),
                'average_price_impact': avg_price_impact
            },
            'trade_characteristics': {
                'total_trades': len(recent_trades),
                'buy_trades': len([t for t in recent_trades if t.side == 'buy']),
                'sell_trades': len([t for t in recent_trades if t.side == 'sell']),
                'average_trade_size': avg_trade_size,
                'median_trade_size': median_trade_size,
                'large_trades_count': len(large_trades),
                'trade_intensity_per_second': trade_intensity
            },
            'large_trades': [
                {
                    'timestamp': trade.timestamp.isoformat(),
                    'side': trade.side,
                    'size': trade.size,
                    'price': trade.price,
                    'size_percentile': stats.percentileofscore(trade_sizes, trade.size)
                }
                for trade in large_trades[:10]  # Top 10 large trades
            ],
            'flow_interpretation': self._interpret_order_flow(flow_imbalance, buy_vwap, sell_vwap),
            'timestamp': datetime.now().isoformat()
        }
    
    def _interpret_order_flow(self, flow_imbalance: float, buy_vwap: float, sell_vwap: float) -> Dict[str, str]:
        """Interpret order flow patterns."""
        
        # Flow imbalance interpretation
        if flow_imbalance > 0.1:
            imbalance_signal = 'Strong buying pressure'
        elif flow_imbalance > 0.05:
            imbalance_signal = 'Moderate buying pressure'
        elif flow_imbalance < -0.1:
            imbalance_signal = 'Strong selling pressure'
        elif flow_imbalance < -0.05:
            imbalance_signal = 'Moderate selling pressure'
        else:
            imbalance_signal = 'Balanced flow'
        
        # VWAP spread interpretation
        vwap_spread_pct = abs(buy_vwap - sell_vwap) / ((buy_vwap + sell_vwap) / 2) * 100 if buy_vwap > 0 and sell_vwap > 0 else 0
        
        if vwap_spread_pct > 0.1:
            vwap_signal = 'High price dispersion between buy and sell orders'
        elif vwap_spread_pct > 0.05:
            vwap_signal = 'Moderate price dispersion'
        else:
            vwap_signal = 'Low price dispersion'
        
        return {
            'flow_imbalance': imbalance_signal,
            'vwap_spread': vwap_signal,
            'overall_signal': f"{imbalance_signal} with {vwap_signal.lower()}"
        }
    
    async def analyze_liquidity(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive liquidity analysis."""
        
        if symbol not in self.orderbook_history:
            return {'error': f'No orderbook data available for {symbol}'}
        
        orderbooks = self.orderbook_history[symbol][-100:]  # Last 100 snapshots
        
        if not orderbooks:
            return {'error': 'No orderbook data found'}
        
        # Calculate various liquidity metrics
        liquidity_metrics = {
            'spread_metrics': self._calculate_spread_metrics(orderbooks),
            'depth_metrics': self._calculate_depth_metrics(orderbooks),
            'resilience_metrics': self._calculate_resilience_metrics(orderbooks),
            'price_impact_metrics': await self._calculate_price_impact_metrics(symbol)
        }
        
        # Calculate composite liquidity score
        composite_score = self._calculate_composite_liquidity_score(liquidity_metrics)
        
        # Identify liquidity patterns
        patterns = self._identify_liquidity_patterns(orderbooks)
        
        return {
            'symbol': symbol,
            'liquidity_metrics': liquidity_metrics,
            'composite_liquidity_score': composite_score,
            'liquidity_patterns': patterns,
            'market_quality_assessment': self._assess_market_quality(liquidity_metrics),
            'trading_cost_estimates': self._estimate_trading_costs(liquidity_metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_spread_metrics(self, orderbooks: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Calculate various spread metrics."""
        
        spreads_absolute = [ob.spread_absolute for ob in orderbooks]
        spreads_percentage = [ob.spread_percentage for ob in orderbooks]
        
        # Effective spreads (weighted by volume)
        effective_spreads = []
        for ob in orderbooks:
            if ob.total_bid_volume > 0 and ob.total_ask_volume > 0:
                bid_weight = ob.bids[0][1] / (ob.bids[0][1] + ob.asks[0][1])
                ask_weight = ob.asks[0][1] / (ob.bids[0][1] + ob.asks[0][1])
                effective_spread = (ob.asks[0][0] - ob.bids[0][0]) * (bid_weight + ask_weight) / 2
                effective_spreads.append(effective_spread)
        
        return {
            'average_spread_absolute': np.mean(spreads_absolute),
            'average_spread_percentage': np.mean(spreads_percentage),
            'median_spread_percentage': np.median(spreads_percentage),
            'spread_volatility': np.std(spreads_percentage),
            'min_spread': min(spreads_percentage),
            'max_spread': max(spreads_percentage),
            'effective_spread': np.mean(effective_spreads) if effective_spreads else 0,
            'spread_percentiles': {
                '25%': np.percentile(spreads_percentage, 25),
                '75%': np.percentile(spreads_percentage, 75),
                '95%': np.percentile(spreads_percentage, 95)
            }
        }
    
    def _calculate_depth_metrics(self, orderbooks: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Calculate market depth metrics."""
        
        # Calculate depth at different price levels
        depth_1pct = []
        depth_5pct = []
        total_volumes = []
        
        for ob in orderbooks:
            mid = ob.mid_price
            
            # 1% depth calculation
            bid_depth_1pct = sum(size for price, size in ob.bids if price >= mid * 0.99)
            ask_depth_1pct = sum(size for price, size in ob.asks if price <= mid * 1.01)
            depth_1pct.append(bid_depth_1pct + ask_depth_1pct)
            
            # 5% depth calculation
            bid_depth_5pct = sum(size for price, size in ob.bids if price >= mid * 0.95)
            ask_depth_5pct = sum(size for price, size in ob.asks if price <= mid * 1.05)
            depth_5pct.append(bid_depth_5pct + ask_depth_5pct)
            
            total_volumes.append(ob.total_bid_volume + ob.total_ask_volume)
        
        # Order book imbalance statistics
        imbalances = [ob.book_imbalance for ob in orderbooks]
        
        return {
            'average_depth_1pct': np.mean(depth_1pct),
            'average_depth_5pct': np.mean(depth_5pct),
            'median_depth_1pct': np.median(depth_1pct),
            'depth_volatility': np.std(depth_1pct),
            'average_total_volume': np.mean(total_volumes),
            'order_book_imbalance': {
                'average': np.mean(imbalances),
                'std': np.std(imbalances),
                'extreme_imbalance_frequency': sum(1 for imb in imbalances if abs(imb) > 0.5) / len(imbalances)
            },
            'depth_concentration': np.mean(depth_1pct) / np.mean(depth_5pct) if np.mean(depth_5pct) > 0 else 0
        }
    
    def _calculate_resilience_metrics(self, orderbooks: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Calculate market resilience metrics."""
        
        # Price reversion analysis
        mid_prices = [ob.mid_price for ob in orderbooks]
        price_changes = np.diff(mid_prices)
        
        # Calculate auto-correlation (measure of mean reversion)
        if len(price_changes) > 10:
            autocorr_1 = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1] if len(price_changes) > 1 else 0
            autocorr_5 = np.corrcoef(price_changes[:-5], price_changes[5:])[0, 1] if len(price_changes) > 5 else 0
        else:
            autocorr_1 = autocorr_5 = 0
        
        # Volatility clustering (ARCH effects)
        squared_returns = price_changes ** 2
        vol_clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] if len(squared_returns) > 1 else 0
        
        # Order book stability
        spread_changes = np.diff([ob.spread_percentage for ob in orderbooks])
        spread_stability = 1 / (1 + np.std(spread_changes)) if len(spread_changes) > 0 else 0
        
        return {
            'price_autocorrelation_1lag': autocorr_1,
            'price_autocorrelation_5lag': autocorr_5,
            'volatility_clustering': vol_clustering,
            'spread_stability': spread_stability,
            'price_volatility': np.std(price_changes) if len(price_changes) > 0 else 0,
            'resilience_score': (1 - abs(autocorr_1)) * spread_stability  # Higher is more resilient
        }
    
    async def _calculate_price_impact_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate price impact metrics from trade data."""
        
        if symbol not in self.trade_history:
            return {}
        
        trades = self.trade_history[symbol][-100:]  # Last 100 trades
        
        if len(trades) < 10:
            return {}
        
        # Calculate Kyle's lambda (price impact coefficient)
        trade_sizes = np.array([t.size for t in trades])
        price_changes = []
        
        for i in range(1, len(trades)):
            price_change = (trades[i].price - trades[i-1].price) / trades[i-1].price
            price_changes.append(price_change)
        
        if len(price_changes) > 0:
            # Simple linear regression for Kyle's lambda
            correlation = np.corrcoef(trade_sizes[1:], price_changes)[0, 1] if len(price_changes) > 1 else 0
            kyle_lambda = correlation * np.std(price_changes) / np.std(trade_sizes[1:]) if np.std(trade_sizes[1:]) > 0 else 0
        else:
            kyle_lambda = 0
        
        # Amihud illiquidity ratio
        daily_returns = np.array(price_changes)
        daily_volumes = trade_sizes[1:]  # Align with price changes
        
        if len(daily_returns) > 0 and len(daily_volumes) > 0:
            amihud_ratio = np.mean(np.abs(daily_returns) / daily_volumes) if np.mean(daily_volumes) > 0 else 0
        else:
            amihud_ratio = 0
        
        return {
            'kyle_lambda': kyle_lambda,
            'amihud_illiquidity': amihud_ratio,
            'average_trade_impact': np.mean(np.abs(price_changes)) if price_changes else 0,
            'impact_volatility': np.std(price_changes) if price_changes else 0
        }
    
    def _calculate_composite_liquidity_score(self, metrics: Dict[str, Dict]) -> float:
        """Calculate composite liquidity score (0-100 scale)."""
        
        score_components = []
        
        # Spread component (lower spreads = higher liquidity)
        if 'spread_metrics' in metrics:
            spread_pct = metrics['spread_metrics'].get('average_spread_percentage', 0)
            spread_score = max(0, 100 - spread_pct * 1000)  # Scale spread to 0-100
            score_components.append(('spread', spread_score, 0.3))
        
        # Depth component (higher depth = higher liquidity)
        if 'depth_metrics' in metrics:
            depth_1pct = metrics['depth_metrics'].get('average_depth_1pct', 0)
            depth_score = min(100, depth_1pct / 1000 * 100)  # Normalize depth
            score_components.append(('depth', depth_score, 0.3))
        
        # Resilience component
        if 'resilience_metrics' in metrics:
            resilience = metrics['resilience_metrics'].get('resilience_score', 0)
            resilience_score = max(0, min(100, resilience * 100))
            score_components.append(('resilience', resilience_score, 0.2))
        
        # Impact component (lower impact = higher liquidity)
        if 'price_impact_metrics' in metrics:
            kyle_lambda = abs(metrics['price_impact_metrics'].get('kyle_lambda', 0))
            impact_score = max(0, 100 - kyle_lambda * 10000)  # Scale impact
            score_components.append(('impact', impact_score, 0.2))
        
        # Calculate weighted score
        if score_components:
            total_weight = sum(weight for _, _, weight in score_components)
            weighted_score = sum(score * weight for _, score, weight in score_components) / total_weight
            
            return {
                'composite_score': weighted_score,
                'components': {name: score for name, score, _ in score_components},
                'interpretation': self._interpret_liquidity_score(weighted_score)
            }
        else:
            return {'composite_score': 0, 'components': {}, 'interpretation': 'Insufficient data'}
    
    def _interpret_liquidity_score(self, score: float) -> str:
        """Interpret composite liquidity score."""
        
        if score >= 80:
            return 'Excellent liquidity - tight spreads, deep markets, resilient'
        elif score >= 60:
            return 'Good liquidity - reasonable spreads and depth'
        elif score >= 40:
            return 'Moderate liquidity - some trading costs expected'
        elif score >= 20:
            return 'Poor liquidity - high spreads and impact costs'
        else:
            return 'Very poor liquidity - significant trading difficulties'
    
    def _identify_liquidity_patterns(self, orderbooks: List[OrderBookSnapshot]) -> Dict[str, Any]:
        """Identify patterns in liquidity provision."""
        
        # Time-based patterns
        hours = [ob.timestamp.hour for ob in orderbooks]
        spreads_by_hour = {}
        volumes_by_hour = {}
        
        for ob in orderbooks:
            hour = ob.timestamp.hour
            if hour not in spreads_by_hour:
                spreads_by_hour[hour] = []
                volumes_by_hour[hour] = []
            
            spreads_by_hour[hour].append(ob.spread_percentage)
            volumes_by_hour[hour].append(ob.total_bid_volume + ob.total_ask_volume)
        
        # Find best and worst liquidity hours
        avg_spreads_by_hour = {hour: np.mean(spreads) for hour, spreads in spreads_by_hour.items()}
        best_liquidity_hour = min(avg_spreads_by_hour.keys(), key=avg_spreads_by_hour.get) if avg_spreads_by_hour else 0
        worst_liquidity_hour = max(avg_spreads_by_hour.keys(), key=avg_spreads_by_hour.get) if avg_spreads_by_hour else 0
        
        # Volatility clustering in spreads
        spread_changes = np.diff([ob.spread_percentage for ob in orderbooks])
        volatility_clustering = np.corrcoef(spread_changes[:-1] ** 2, spread_changes[1:] ** 2)[0, 1] if len(spread_changes) > 1 else 0
        
        # Identify regime changes (using DBSCAN clustering)
        if len(orderbooks) >= 20:
            features = np.column_stack([
                [ob.spread_percentage for ob in orderbooks],
                [ob.total_bid_volume + ob.total_ask_volume for ob in orderbooks],
                [ob.book_imbalance for ob in orderbooks]
            ])
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            clustering = DBSCAN(eps=0.5, min_samples=5)
            clusters = clustering.fit_predict(features_scaled)
            n_regimes = len(set(clusters)) - (1 if -1 in clusters else 0)
        else:
            n_regimes = 1
        
        return {
            'temporal_patterns': {
                'best_liquidity_hour': best_liquidity_hour,
                'worst_liquidity_hour': worst_liquidity_hour,
                'hourly_spread_variation': np.std(list(avg_spreads_by_hour.values())) if avg_spreads_by_hour else 0
            },
            'volatility_patterns': {
                'spread_volatility_clustering': volatility_clustering,
                'high_volatility_periods': sum(1 for change in spread_changes if abs(change) > np.std(spread_changes) * 2) / len(spread_changes) if len(spread_changes) > 0 else 0
            },
            'market_regimes': {
                'number_of_regimes': n_regimes,
                'regime_stability': 1 / n_regimes if n_regimes > 0 else 1
            }
        }
    
    def _assess_market_quality(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall market quality."""
        
        assessments = {}
        
        # Spread assessment
        if 'spread_metrics' in metrics:
            avg_spread = metrics['spread_metrics'].get('average_spread_percentage', 0)
            if avg_spread < 0.05:
                assessments['spread_quality'] = 'Excellent (< 0.05%)'
            elif avg_spread < 0.1:
                assessments['spread_quality'] = 'Good (< 0.1%)'
            elif avg_spread < 0.2:
                assessments['spread_quality'] = 'Fair (< 0.2%)'
            else:
                assessments['spread_quality'] = 'Poor (> 0.2%)'
        
        # Depth assessment
        if 'depth_metrics' in metrics:
            imbalance_volatility = metrics['depth_metrics']['order_book_imbalance'].get('std', 0)
            if imbalance_volatility < 0.1:
                assessments['depth_quality'] = 'Stable and balanced'
            elif imbalance_volatility < 0.2:
                assessments['depth_quality'] = 'Moderately stable'
            else:
                assessments['depth_quality'] = 'Unstable with frequent imbalances'
        
        # Market resilience
        if 'resilience_metrics' in metrics:
            resilience = metrics['resilience_metrics'].get('resilience_score', 0)
            if resilience > 0.8:
                assessments['resilience_quality'] = 'Highly resilient market'
            elif resilience > 0.6:
                assessments['resilience_quality'] = 'Good resilience'
            elif resilience > 0.4:
                assessments['resilience_quality'] = 'Moderate resilience'
            else:
                assessments['resilience_quality'] = 'Low resilience - prone to price gaps'
        
        return assessments
    
    def _estimate_trading_costs(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Estimate trading costs for different order sizes."""
        
        costs = {}
        
        if 'spread_metrics' in metrics and 'price_impact_metrics' in metrics:
            spread_cost = metrics['spread_metrics'].get('effective_spread', 0) / 2  # Half spread cost
            impact_coeff = metrics['price_impact_metrics'].get('kyle_lambda', 0)
            
            # Estimate costs for different trade sizes (as % of trade value)
            trade_sizes = [1000, 10000, 100000, 1000000]  # USD values
            
            for size in trade_sizes:
                # Total cost = spread cost + market impact
                impact_cost = abs(impact_coeff) * np.sqrt(size / 10000)  # Square root impact model
                total_cost = spread_cost + impact_cost
                
                costs[f'size_{size}'] = {
                    'spread_cost_pct': spread_cost * 100,
                    'impact_cost_pct': impact_cost * 100,
                    'total_cost_pct': total_cost * 100
                }
        
        return costs
    
    def get_microstructure_summary(self) -> Dict[str, Any]:
        """Get comprehensive market microstructure summary."""
        
        return {
            'system_status': {
                'symbols_tracked': len(self.orderbook_history),
                'orderbook_snapshots': sum(len(obs) for obs in self.orderbook_history.values()),
                'trade_records': sum(len(trades) for trades in self.trade_history.values()),
                'models_initialized': len(self.impact_models) > 0
            },
            'supported_analysis': [
                'Order flow analysis and imbalance detection',
                'Multi-dimensional liquidity measurement',
                'Market impact modeling',
                'Spread decomposition analysis',
                'Market resilience assessment',
                'Trading cost estimation',
                'Liquidity pattern identification',
                'Market regime detection'
            ],
            'tracked_symbols': list(self.orderbook_history.keys()),
            'analysis_parameters': {
                'orderbook_depth_levels': self.orderbook_depth,
                'trade_analysis_window': self.tick_analysis_window,
                'impact_measurement_window_seconds': self.impact_measurement_window
            },
            'model_types': {
                'liquidity_models': list(self.liquidity_models.keys()) if hasattr(self, 'liquidity_models') else [],
                'impact_models': list(self.impact_models.keys())
            },
            'timestamp': datetime.now().isoformat()
        }

# Global microstructure analyzer instance
microstructure_analyzer = MarketMicrostructureAnalyzer()