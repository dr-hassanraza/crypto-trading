"""
Simplified Cryptocurrency API Integration
=========================================

This is a simplified version that makes minimal API calls to avoid rate limiting.
Uses aggressive caching and batch requests to minimize CoinGecko API usage.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import streamlit as st


class SimpleCryptoAPI:
    """Simplified API manager with aggressive caching and minimal requests."""
    
    def __init__(self):
        self.base_url = 'https://api.coingecko.com/api/v3'
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StreamlitCryptoApp/1.0',
            'Accept': 'application/json'
        })
        
        # Track last request time
        self.last_request_time = 0
        self.min_interval = 3.0  # 3 seconds between requests
        
        # Cache storage
        self._cache = {}
        self._cache_timestamps = {}

    def _should_use_cache(self, key: str, ttl: int) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache or key not in self._cache_timestamps:
            return False
        
        age = time.time() - self._cache_timestamps[key]
        return age < ttl

    def _get_from_cache(self, key: str):
        """Get data from cache."""
        return self._cache.get(key)

    def _set_cache(self, key: str, data: Any):
        """Store data in cache."""
        self._cache[key] = data
        self._cache_timestamps[key] = time.time()

    def _make_request(self, endpoint: str, params: Dict = None, cache_key: str = None, ttl: int = 600) -> Optional[Dict]:
        """Make API request with caching and rate limiting."""
        # Check cache first
        if cache_key and self._should_use_cache(cache_key, ttl):
            return self._get_from_cache(cache_key)
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            st.info(f"⏳ Rate limiting - waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 429:
                st.error("🚫 Rate limit exceeded - using cached/fallback data")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            if cache_key:
                self._set_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            st.warning(f"API error: {str(e)}")
            return None

    def get_simple_prices(self) -> Dict[str, Any]:
        """Get basic price data for major cryptocurrencies."""
        params = {
            'ids': 'bitcoin,ethereum,cardano,solana',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_market_cap': 'true'
        }
        
        data = self._make_request(
            'simple/price', 
            params=params, 
            cache_key='simple_prices',
            ttl=300  # 5 minutes cache
        )
        
        if not data:
            # Fallback data
            return self._get_fallback_prices()
        
        return data

    def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency market data."""
        data = self._make_request(
            'global', 
            cache_key='global_data',
            ttl=1800  # 30 minutes cache
        )
        
        if not data:
            return {
                'total_market_cap': {'usd': 2000000000000},
                'total_volume': {'usd': 50000000000},
                'market_cap_percentage': {'btc': 45, 'eth': 18}
            }
        
        return data.get('data', {})

    def get_basic_historical_data(self, coin_id: str, days: int = 7) -> List[List]:
        """Get basic historical data (prices only) for reduced API usage."""
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily'
        }
        
        data = self._make_request(
            f'coins/{coin_id}/market_chart',
            params=params,
            cache_key=f'history_{coin_id}_{days}',
            ttl=3600  # 1 hour cache
        )
        
        if not data or 'prices' not in data:
            return self._generate_fallback_historical(coin_id, days)
        
        return data['prices']

    def _get_fallback_prices(self) -> Dict[str, Any]:
        """Generate fallback price data."""
        return {
            'bitcoin': {
                'usd': 45000 + np.random.normal(0, 2000),
                'usd_24h_change': np.random.uniform(-5, 5),
                'usd_market_cap': 900000000000
            },
            'ethereum': {
                'usd': 2500 + np.random.normal(0, 200),
                'usd_24h_change': np.random.uniform(-5, 5),
                'usd_market_cap': 300000000000
            },
            'cardano': {
                'usd': 0.5 + np.random.normal(0, 0.05),
                'usd_24h_change': np.random.uniform(-5, 5),
                'usd_market_cap': 15000000000
            },
            'solana': {
                'usd': 100 + np.random.normal(0, 10),
                'usd_24h_change': np.random.uniform(-5, 5),
                'usd_market_cap': 25000000000
            }
        }

    def _generate_fallback_historical(self, coin_id: str, days: int) -> List[List]:
        """Generate fallback historical data."""
        base_prices = {
            'bitcoin': 45000,
            'ethereum': 2500,
            'cardano': 0.5,
            'solana': 100
        }
        
        base_price = base_prices.get(coin_id, 100)
        prices = []
        
        for i in range(days):
            timestamp = int((datetime.now() - timedelta(days=days-i)).timestamp() * 1000)
            price = base_price * (1 + np.random.normal(0, 0.02))
            prices.append([timestamp, price])
        
        return prices

    def get_trending_simple(self) -> List[str]:
        """Get just trending coin names to minimize data."""
        data = self._make_request(
            'search/trending',
            cache_key='trending_simple',
            ttl=3600  # 1 hour cache
        )
        
        if not data or 'coins' not in data:
            return ['Bitcoin', 'Ethereum', 'Cardano', 'Solana', 'Polygon']
        
        return [coin['item']['name'] for coin in data['coins'][:5]]


# Create simplified global instance
simple_crypto_api = SimpleCryptoAPI()


# Simplified convenience functions
@st.cache_data(ttl=600)
def get_live_prices():
    """Get live cryptocurrency prices with heavy caching."""
    return simple_crypto_api.get_simple_prices()

@st.cache_data(ttl=1800)
def get_global_market_data():
    """Get global market data with heavy caching."""
    return simple_crypto_api.get_global_data()

@st.cache_data(ttl=3600)
def get_price_history(coin_id: str, days: int = 7):
    """Get price history with heavy caching."""
    return simple_crypto_api.get_basic_historical_data(coin_id, days)

@st.cache_data(ttl=3600)
def get_trending_coins():
    """Get trending coins with heavy caching."""
    return simple_crypto_api.get_trending_simple()


def convert_to_dataframe(price_history: List[List], symbol: str) -> pd.DataFrame:
    """Convert price history to DataFrame for charting."""
    if not price_history:
        # Generate fallback data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        base_price = {'bitcoin': 45000, 'ethereum': 2500, 'cardano': 0.5, 'solana': 100}.get(symbol, 100)
        prices = [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(7)]
        
        return pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Volume': [1000000 * np.random.uniform(0.5, 2) for _ in range(7)]
        })
    
    # Convert API data
    dates = [datetime.fromtimestamp(item[0] / 1000) for item in price_history]
    closes = [item[1] for item in price_history]
    
    # Generate OHLV from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, closes)):
        volatility = 0.02
        open_price = closes[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + volatility/2)
        low = min(open_price, close) * (1 - volatility/2)
        volume = 1000000 * np.random.uniform(0.5, 2)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the simplified API
    print("Testing Simplified Crypto API...")
    
    prices = get_live_prices()
    print(f"Live Prices: {len(prices)} coins")
    
    history = get_price_history('bitcoin', 7)
    print(f"BTC History: {len(history)} data points")
    
    trending = get_trending_coins()
    print(f"Trending: {trending}")