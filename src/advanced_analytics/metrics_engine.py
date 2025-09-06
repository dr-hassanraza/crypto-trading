"""
Advanced Metrics Engine API Integration

High-performance metrics calculation and API integration system for
real-time market analysis with sub-500ms response times.
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async
from src.data_sources.market_data import MarketDataFetcher


@dataclass
class MetricsConfig:
    """Configuration for metrics engine."""
    
    # Performance settings
    max_response_time_ms: int = 200  # API response target
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 30
    
    # Data sources
    primary_data_source: str = "coingecko"  # coingecko, binance, etc.
    fallback_data_sources: List[str] = field(default_factory=lambda: ["binance"])
    
    # Metrics calculation
    enable_realtime_metrics: bool = True
    enable_technical_metrics: bool = True
    enable_volume_metrics: bool = True
    enable_volatility_metrics: bool = True
    enable_correlation_metrics: bool = True
    
    # API endpoints
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"
    binance_api_url: str = "https://api.binance.com/api/v3"
    
    # Rate limiting
    requests_per_minute: int = 50
    enable_rate_limiting: bool = True


@dataclass
class RealTimeMetrics:
    """Real-time market metrics."""
    symbol: str
    timestamp: datetime
    
    # Price metrics
    current_price: float
    price_change_24h: float
    price_change_pct_24h: float
    high_24h: float
    low_24h: float
    
    # Volume metrics
    volume_24h: float
    volume_change_pct_24h: float
    market_cap: Optional[float] = None
    
    # Technical metrics
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    # Volatility metrics
    volatility_1h: Optional[float] = None
    volatility_24h: Optional[float] = None
    
    # Market strength metrics
    buying_pressure: Optional[float] = None
    selling_pressure: Optional[float] = None
    
    # Performance metrics
    fetch_time_ms: float = 0.0


@dataclass
class MarketState:
    """Current market state aggregation."""
    timestamp: datetime
    overall_sentiment: str  # bullish, bearish, neutral
    volatility_regime: str  # low, medium, high
    liquidity_score: float
    risk_level: str  # low, medium, high
    opportunity_score: float
    dominant_trend: str  # up, down, sideways
    correlation_breakdown: Dict[str, float] = field(default_factory=dict)


class HighPerformanceAPIClient:
    """High-performance async API client with intelligent caching."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.cache = {}
        self.rate_limiter = {}
        
        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'CryptoTrendAnalyzer/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_market_data(self, symbol: str, source: str = None) -> Dict[str, Any]:
        """Fetch market data with intelligent caching and fallbacks."""
        source = source or self.config.primary_data_source
        cache_key = f"{source}:{symbol}"
        
        # Check cache first
        if self.config.enable_caching and cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.config.cache_ttl_seconds:
                self.request_stats['cache_hits'] += 1
                return cached_data
        
        # Rate limiting
        if self.config.enable_rate_limiting:
            await self._enforce_rate_limit(source)
        
        start_time = time.time()
        
        try:
            async with self.request_semaphore:
                data = await self._fetch_from_source(symbol, source)
                
                # Cache successful response
                if self.config.enable_caching:
                    self.cache[cache_key] = (data, datetime.now())
                
                # Update stats
                response_time = (time.time() - start_time) * 1000
                self._update_request_stats(response_time, success=True)
                
                return data
                
        except Exception as e:
            # Try fallback sources
            for fallback_source in self.config.fallback_data_sources:
                if fallback_source != source:
                    try:
                        crypto_logger.logger.warning(
                            f"Primary source {source} failed for {symbol}, trying {fallback_source}"
                        )
                        data = await self._fetch_from_source(symbol, fallback_source)
                        
                        # Cache fallback response
                        if self.config.enable_caching:
                            self.cache[cache_key] = (data, datetime.now())
                        
                        response_time = (time.time() - start_time) * 1000
                        self._update_request_stats(response_time, success=True)
                        
                        return data
                        
                    except Exception as fallback_error:
                        crypto_logger.logger.warning(f"Fallback {fallback_source} also failed: {fallback_error}")
                        continue
            
            # All sources failed
            self._update_request_stats(0, success=False)
            await handle_error_async(e, {'component': 'metrics_api_client', 'symbol': symbol})
            return {}
    
    async def _fetch_from_source(self, symbol: str, source: str) -> Dict[str, Any]:
        """Fetch data from specific source."""
        if source == "coingecko":
            return await self._fetch_coingecko_data(symbol)
        elif source == "binance":
            return await self._fetch_binance_data(symbol)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    async def _fetch_coingecko_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from CoinGecko API."""
        # Convert symbol to CoinGecko format
        coin_id = symbol.lower().replace('usdt', '').replace('btc', 'bitcoin')
        
        url = f"{self.config.coingecko_api_url}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_coingecko_response(data)
            else:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )
    
    async def _fetch_binance_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Binance API."""
        url = f"{self.config.binance_api_url}/ticker/24hr"
        params = {'symbol': symbol.upper()}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_binance_response(data, symbol)
            else:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )
    
    def _parse_coingecko_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CoinGecko API response."""
        try:
            market_data = data.get('market_data', {})
            
            return {
                'symbol': data.get('symbol', '').upper(),
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'price_change_pct_24h': market_data.get('price_change_percentage_24h', 0),
                'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'source': 'coingecko'
            }
        except Exception as e:
            crypto_logger.logger.error(f"Failed to parse CoinGecko response: {e}")
            return {}
    
    def _parse_binance_response(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse Binance API response."""
        try:
            return {
                'symbol': symbol.upper(),
                'current_price': float(data.get('lastPrice', 0)),
                'price_change_24h': float(data.get('priceChange', 0)),
                'price_change_pct_24h': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'volume_24h': float(data.get('volume', 0)),
                'market_cap': None,  # Not available in Binance ticker
                'source': 'binance'
            }
        except Exception as e:
            crypto_logger.logger.error(f"Failed to parse Binance response: {e}")
            return {}
    
    async def _enforce_rate_limit(self, source: str):
        """Enforce rate limiting per source."""
        current_time = datetime.now()
        
        if source not in self.rate_limiter:
            self.rate_limiter[source] = []
        
        # Clean old requests
        minute_ago = current_time - timedelta(minutes=1)
        self.rate_limiter[source] = [
            req_time for req_time in self.rate_limiter[source]
            if req_time > minute_ago
        ]
        
        # Check if we need to wait
        if len(self.rate_limiter[source]) >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - self.rate_limiter[source][0]).seconds
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.rate_limiter[source].append(current_time)
    
    def _update_request_stats(self, response_time_ms: float, success: bool):
        """Update request statistics."""
        self.request_stats['total_requests'] += 1
        
        if success:
            # Update rolling average
            current_avg = self.request_stats['avg_response_time_ms']
            total_requests = self.request_stats['total_requests']
            
            self.request_stats['avg_response_time_ms'] = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
        else:
            self.request_stats['failed_requests'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API client performance statistics."""
        total = self.request_stats['total_requests']
        cache_hits = self.request_stats['cache_hits']
        failed = self.request_stats['failed_requests']
        
        return {
            'total_requests': total,
            'cache_hit_rate': cache_hits / max(total, 1),
            'error_rate': failed / max(total, 1),
            'avg_response_time_ms': self.request_stats['avg_response_time_ms'],
            'cache_size': len(self.cache)
        }


class TechnicalMetricsCalculator:
    """High-performance technical metrics calculation."""
    
    def __init__(self):
        self.calculation_cache = {}
    
    def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators with caching."""
        if ohlcv_data.empty:
            return {}
        
        try:
            # Use hash for caching
            data_hash = hash(str(ohlcv_data.iloc[-20:].values.tobytes()))
            
            if data_hash in self.calculation_cache:
                return self.calculation_cache[data_hash]
            
            indicators = {}
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            volume = ohlcv_data.get('volume', pd.Series())
            
            # RSI (optimized calculation)
            if len(close) >= 14:
                indicators['rsi'] = self._calculate_rsi(close, 14)
            
            # MACD
            if len(close) >= 26:
                macd_line, signal_line = self._calculate_macd(close)
                indicators['macd'] = macd_line
                indicators['macd_signal'] = signal_line
            
            # Bollinger Bands position
            if len(close) >= 20:
                bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
                current_price = close.iloc[-1]
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    indicators['bollinger_position'] = (current_price - bb_lower) / bb_range
            
            # Volume indicators
            if not volume.empty and len(volume) >= 20:
                volume_sma = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                if volume_sma > 0:
                    indicators['volume_ratio'] = current_volume / volume_sma
            
            # Volatility
            if len(close) >= 20:
                returns = close.pct_change()
                indicators['volatility_20'] = returns.rolling(20).std().iloc[-1] * np.sqrt(24 * 365)
            
            # Cache result
            self.calculation_cache[data_hash] = indicators
            
            # Limit cache size
            if len(self.calculation_cache) > 100:
                oldest_key = min(self.calculation_cache.keys())
                del self.calculation_cache[oldest_key]
            
            return indicators
            
        except Exception as e:
            crypto_logger.logger.warning(f"Technical indicator calculation failed: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI efficiently."""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window).mean().iloc[-1]
            avg_loss = loss.rolling(window).mean().iloc[-1]
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception:
            return 50.0  # Neutral RSI on error
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD efficiently."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands efficiently."""
        try:
            rolling_mean = prices.rolling(window).mean()
            rolling_std = prices.rolling(window).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return float(upper_band.iloc[-1]), float(lower_band.iloc[-1])
            
        except Exception:
            current_price = float(prices.iloc[-1])
            return current_price * 1.02, current_price * 0.98  # 2% bands on error


class AdvancedMetricsEngine:
    """Main metrics engine with API integration and performance optimization."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.api_client = HighPerformanceAPIClient(self.config)
        self.tech_calculator = TechnicalMetricsCalculator()
        
        self.metrics_cache = {}
        self.performance_history = []
        
        crypto_logger.logger.info("📊 Advanced Metrics Engine initialized")
    
    async def get_realtime_metrics(self, symbol: str) -> RealTimeMetrics:
        """Get comprehensive real-time metrics for a symbol."""
        start_time = time.time()
        
        try:
            async with HighPerformanceAPIClient(self.config) as client:
                # Fetch basic market data
                market_data = await client.fetch_market_data(symbol)
                
                if not market_data:
                    return self._create_empty_metrics(symbol, start_time)
                
                # Get historical data for technical calculations
                historical_data = await self._get_historical_ohlcv(symbol)
                
                # Calculate technical indicators
                technical_indicators = {}
                if not historical_data.empty and self.config.enable_technical_metrics:
                    technical_indicators = self.tech_calculator.calculate_technical_indicators(historical_data)
                
                # Calculate volatility metrics
                volatility_metrics = {}
                if not historical_data.empty and self.config.enable_volatility_metrics:
                    volatility_metrics = self._calculate_volatility_metrics(historical_data)
                
                # Create metrics object
                fetch_time = (time.time() - start_time) * 1000
                
                metrics = RealTimeMetrics(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    current_price=market_data.get('current_price', 0),
                    price_change_24h=market_data.get('price_change_24h', 0),
                    price_change_pct_24h=market_data.get('price_change_pct_24h', 0),
                    high_24h=market_data.get('high_24h', 0),
                    low_24h=market_data.get('low_24h', 0),
                    volume_24h=market_data.get('volume_24h', 0),
                    market_cap=market_data.get('market_cap'),
                    rsi=technical_indicators.get('rsi'),
                    macd=technical_indicators.get('macd'),
                    bollinger_position=technical_indicators.get('bollinger_position'),
                    volatility_24h=volatility_metrics.get('volatility_24h'),
                    fetch_time_ms=fetch_time
                )
                
                # Performance tracking
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'fetch_time_ms': fetch_time,
                    'success': True
                })
                
                # Check performance target
                if fetch_time > self.config.max_response_time_ms:
                    crypto_logger.logger.warning(
                        f"Metrics fetch for {symbol} took {fetch_time:.2f}ms, "
                        f"exceeds target {self.config.max_response_time_ms}ms"
                    )
                
                crypto_logger.logger.debug(f"Metrics for {symbol} fetched in {fetch_time:.2f}ms")
                
                return metrics
                
        except Exception as e:
            await handle_error_async(e, {'component': 'metrics_engine', 'symbol': symbol})
            return self._create_empty_metrics(symbol, start_time)
    
    async def get_market_state(self, symbols: List[str]) -> MarketState:
        """Get overall market state from multiple symbols."""
        try:
            # Fetch metrics for all symbols concurrently
            tasks = [self.get_realtime_metrics(symbol) for symbol in symbols[:10]]  # Limit for performance
            metrics_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            valid_metrics = [m for m in metrics_list if isinstance(m, RealTimeMetrics)]
            
            if not valid_metrics:
                return MarketState(
                    timestamp=datetime.now(),
                    overall_sentiment='neutral',
                    volatility_regime='medium',
                    liquidity_score=0.5,
                    risk_level='medium',
                    opportunity_score=0.5,
                    dominant_trend='sideways'
                )
            
            # Analyze market state
            sentiment = self._analyze_market_sentiment(valid_metrics)
            volatility_regime = self._analyze_volatility_regime(valid_metrics)
            liquidity_score = self._calculate_liquidity_score(valid_metrics)
            risk_level = self._assess_risk_level(valid_metrics)
            opportunity_score = self._calculate_opportunity_score(valid_metrics)
            dominant_trend = self._identify_dominant_trend(valid_metrics)
            
            return MarketState(
                timestamp=datetime.now(),
                overall_sentiment=sentiment,
                volatility_regime=volatility_regime,
                liquidity_score=liquidity_score,
                risk_level=risk_level,
                opportunity_score=opportunity_score,
                dominant_trend=dominant_trend
            )
            
        except Exception as e:
            await handle_error_async(e, {'component': 'market_state_analysis'})
            return MarketState(
                timestamp=datetime.now(),
                overall_sentiment='neutral',
                volatility_regime='medium',
                liquidity_score=0.5,
                risk_level='high',  # Conservative on error
                opportunity_score=0.0,
                dominant_trend='sideways'
            )
    
    async def _get_historical_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data for technical analysis."""
        try:
            # This would integrate with your existing market data fetcher
            # For now, return empty DataFrame
            return pd.DataFrame()
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_volatility_metrics(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics."""
        try:
            if ohlcv_data.empty or 'close' not in ohlcv_data.columns:
                return {}
            
            returns = ohlcv_data['close'].pct_change().dropna()
            
            if len(returns) < 24:
                return {}
            
            # 24-hour volatility (annualized)
            vol_24h = returns.tail(24).std() * np.sqrt(24 * 365)
            
            return {
                'volatility_24h': float(vol_24h)
            }
            
        except Exception:
            return {}
    
    def _analyze_market_sentiment(self, metrics: List[RealTimeMetrics]) -> str:
        """Analyze overall market sentiment."""
        positive_changes = sum(1 for m in metrics if m.price_change_pct_24h > 0)
        total_metrics = len(metrics)
        
        if total_metrics == 0:
            return 'neutral'
        
        positive_ratio = positive_changes / total_metrics
        
        if positive_ratio > 0.6:
            return 'bullish'
        elif positive_ratio < 0.4:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_volatility_regime(self, metrics: List[RealTimeMetrics]) -> str:
        """Analyze volatility regime."""
        volatilities = [abs(m.price_change_pct_24h) for m in metrics if m.price_change_pct_24h is not None]
        
        if not volatilities:
            return 'medium'
        
        avg_volatility = np.mean(volatilities)
        
        if avg_volatility < 2:
            return 'low'
        elif avg_volatility > 5:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_liquidity_score(self, metrics: List[RealTimeMetrics]) -> float:
        """Calculate liquidity score (0-1)."""
        volumes = [m.volume_24h for m in metrics if m.volume_24h > 0]
        
        if not volumes:
            return 0.5
        
        # Normalized liquidity score based on volume
        avg_volume = np.mean(volumes)
        
        # Simple heuristic: higher volume = higher liquidity
        score = min(1.0, avg_volume / 1000000000)  # $1B volume = perfect liquidity
        
        return float(score)
    
    def _assess_risk_level(self, metrics: List[RealTimeMetrics]) -> str:
        """Assess overall risk level."""
        # Simple risk assessment based on volatility
        volatilities = [abs(m.price_change_pct_24h) for m in metrics if m.price_change_pct_24h is not None]
        
        if not volatilities:
            return 'medium'
        
        max_volatility = max(volatilities)
        
        if max_volatility > 10:
            return 'high'
        elif max_volatility < 3:
            return 'low'
        else:
            return 'medium'
    
    def _calculate_opportunity_score(self, metrics: List[RealTimeMetrics]) -> float:
        """Calculate opportunity score (0-1)."""
        # Combine volatility and volume for opportunity
        volatilities = [abs(m.price_change_pct_24h) for m in metrics if m.price_change_pct_24h is not None]
        volumes = [m.volume_24h for m in metrics if m.volume_24h > 0]
        
        if not volatilities or not volumes:
            return 0.5
        
        # Moderate volatility + high volume = high opportunity
        avg_volatility = np.mean(volatilities)
        avg_volume = np.mean(volumes)
        
        volatility_score = min(1.0, avg_volatility / 5.0)  # 5% = perfect volatility
        volume_score = min(1.0, avg_volume / 1000000000)  # $1B = perfect volume
        
        opportunity_score = (volatility_score + volume_score) / 2
        
        return float(opportunity_score)
    
    def _identify_dominant_trend(self, metrics: List[RealTimeMetrics]) -> str:
        """Identify dominant market trend."""
        price_changes = [m.price_change_pct_24h for m in metrics if m.price_change_pct_24h is not None]
        
        if not price_changes:
            return 'sideways'
        
        avg_change = np.mean(price_changes)
        
        if avg_change > 2:
            return 'up'
        elif avg_change < -2:
            return 'down'
        else:
            return 'sideways'
    
    def _create_empty_metrics(self, symbol: str, start_time: float) -> RealTimeMetrics:
        """Create empty metrics for error cases."""
        return RealTimeMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=0.0,
            price_change_24h=0.0,
            price_change_pct_24h=0.0,
            high_24h=0.0,
            low_24h=0.0,
            volume_24h=0.0,
            fetch_time_ms=(time.time() - start_time) * 1000
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get metrics engine performance statistics."""
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-50:]  # Last 50 requests
        fetch_times = [h['fetch_time_ms'] for h in recent_history]
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        
        return {
            'avg_fetch_time_ms': np.mean(fetch_times),
            'max_fetch_time_ms': np.max(fetch_times),
            'min_fetch_time_ms': np.min(fetch_times),
            'success_rate': success_rate,
            'performance_target_met': np.mean(fetch_times) < self.config.max_response_time_ms,
            'total_requests': len(self.performance_history),
            'cache_size': len(self.metrics_cache)
        }
    
    def update_config(self, new_config: MetricsConfig):
        """Update metrics engine configuration."""
        self.config = new_config
        self.api_client = HighPerformanceAPIClient(new_config)
        crypto_logger.logger.info("Metrics engine configuration updated")


# Global metrics engine instance
metrics_engine = AdvancedMetricsEngine()