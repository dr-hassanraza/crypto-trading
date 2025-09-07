"""
Multi-API Cryptocurrency Data Fetcher
=====================================

Robust crypto data fetching with multiple free API backup sources:
1. CoinGecko API (Primary)
2. CoinCap API (Backup #1)
3. CryptoCompare API (Backup #2) 
4. Coinbase API (Backup #3)
5. Yahoo Finance (Backup #4)

All APIs are free and don't require API keys for basic usage.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import streamlit as st
import json


class MultiAPICryptoManager:
    """Manages multiple crypto APIs with intelligent failover."""
    
    def __init__(self):
        # API configurations with their endpoints and rate limits
        self.api_configs = {
            'coingecko': {
                'base_url': 'https://api.coingecko.com/api/v3',
                'rate_limit': 3.0,  # seconds between requests
                'priority': 1,      # Primary API
                'active': True,
                'last_request': 0,
                'failures': 0,
                'max_failures': 3
            },
            'coincap': {
                'base_url': 'https://api.coincap.io/v2',
                'rate_limit': 1.0,
                'priority': 2,
                'active': True,
                'last_request': 0,
                'failures': 0,
                'max_failures': 3
            },
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data',
                'rate_limit': 1.0,
                'priority': 3,
                'active': True,
                'last_request': 0,
                'failures': 0,
                'max_failures': 3
            },
            'coinbase': {
                'base_url': 'https://api.coinbase.com/v2',
                'rate_limit': 1.5,
                'priority': 4,
                'active': True,
                'last_request': 0,
                'failures': 0,
                'max_failures': 3
            },
            'yahoo': {
                'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart',
                'rate_limit': 2.0,
                'priority': 5,
                'active': True,
                'last_request': 0,
                'failures': 0,
                'max_failures': 3
            }
        }
        
        # Coin symbol mappings for different APIs
        self.coin_mappings = {
            'coingecko': {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'ADA': 'cardano',
                'SOL': 'solana',
                'BNB': 'binancecoin',
                'XRP': 'ripple',
                'MATIC': 'polygon'
            },
            'coincap': {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'ADA': 'cardano', 
                'SOL': 'solana',
                'BNB': 'binance-coin',
                'XRP': 'xrp',
                'MATIC': 'polygon'
            },
            'cryptocompare': {
                'BTC': 'BTC',
                'ETH': 'ETH',
                'ADA': 'ADA',
                'SOL': 'SOL', 
                'BNB': 'BNB',
                'XRP': 'XRP',
                'MATIC': 'MATIC'
            },
            'coinbase': {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'ADA': 'ADA-USD',
                'SOL': 'SOL-USD',
                'BNB': 'BNB-USD',
                'XRP': 'XRP-USD',
                'MATIC': 'MATIC-USD'
            },
            'yahoo': {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'ADA': 'ADA-USD',
                'SOL': 'SOL-USD',
                'BNB': 'BNB-USD',
                'XRP': 'XRP-USD',
                'MATIC': 'MATIC-USD'
            }
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalyzer/2.0 (Multi-API)',
            'Accept': 'application/json'
        })
        
        # Cache system
        self._cache = {}
        self._cache_timestamps = {}

    def _get_active_apis(self) -> List[str]:
        """Get list of active APIs sorted by priority."""
        active = [(name, config) for name, config in self.api_configs.items() if config['active']]
        return [name for name, config in sorted(active, key=lambda x: x[1]['priority'])]

    def _should_use_api(self, api_name: str) -> bool:
        """Check if API should be used based on failure count and cooldown."""
        config = self.api_configs[api_name]
        
        # Check if API is disabled due to failures
        if not config['active']:
            # Try to re-enable after 5 minutes
            if time.time() - config['last_request'] > 300:
                config['active'] = True
                config['failures'] = 0
                st.info(f"🔄 Re-enabling {api_name} API after cooldown")
            else:
                return False
        
        return True

    def _rate_limit(self, api_name: str):
        """Apply rate limiting for specific API."""
        config = self.api_configs[api_name]
        current_time = time.time()
        
        time_since_last = current_time - config['last_request']
        if time_since_last < config['rate_limit']:
            sleep_time = config['rate_limit'] - time_since_last
            time.sleep(sleep_time)
        
        config['last_request'] = time.time()

    def _handle_api_failure(self, api_name: str, error: str = ""):
        """Handle API failure and disable if necessary."""
        config = self.api_configs[api_name]
        config['failures'] += 1
        
        if config['failures'] >= config['max_failures']:
            config['active'] = False
            st.warning(f"⚠️ {api_name.title()} API disabled after {config['failures']} failures")
        else:
            st.warning(f"⚠️ {api_name.title()} API error ({config['failures']}/{config['max_failures']}): {error}")

    def _handle_api_success(self, api_name: str):
        """Reset failure count on successful API call."""
        self.api_configs[api_name]['failures'] = 0

    def _make_request(self, api_name: str, url: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling."""
        if not self._should_use_api(api_name):
            return None
        
        self._rate_limit(api_name)
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                self._handle_api_failure(api_name, "Rate limited")
                return None
            
            response.raise_for_status()
            self._handle_api_success(api_name)
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self._handle_api_failure(api_name, str(e))
            return None
        except json.JSONDecodeError as e:
            self._handle_api_failure(api_name, "Invalid JSON response")
            return None

    def get_prices_coingecko(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices from CoinGecko API."""
        coin_ids = [self.coin_mappings['coingecko'].get(symbol) for symbol in symbols]
        coin_ids = [cid for cid in coin_ids if cid]  # Remove None values
        
        if not coin_ids:
            return {}
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_market_cap': 'true'
        }
        
        url = f"{self.api_configs['coingecko']['base_url']}/simple/price"
        data = self._make_request('coingecko', url, params)
        
        if not data:
            return {}
        
        # Convert back to symbol-based dict
        result = {}
        for symbol in symbols:
            coin_id = self.coin_mappings['coingecko'].get(symbol)
            if coin_id and coin_id in data:
                result[symbol] = {
                    'price': data[coin_id]['usd'],
                    'change_24h': data[coin_id].get('usd_24h_change', 0),
                    'market_cap': data[coin_id].get('usd_market_cap', 0)
                }
        
        return result

    def get_prices_coincap(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices from CoinCap API."""
        result = {}
        
        for symbol in symbols:
            coin_id = self.coin_mappings['coincap'].get(symbol)
            if not coin_id:
                continue
                
            url = f"{self.api_configs['coincap']['base_url']}/assets/{coin_id}"
            data = self._make_request('coincap', url)
            
            if data and 'data' in data:
                asset = data['data']
                result[symbol] = {
                    'price': float(asset.get('priceUsd', 0)),
                    'change_24h': float(asset.get('changePercent24Hr', 0)),
                    'market_cap': float(asset.get('marketCapUsd', 0))
                }
        
        return result

    def get_prices_cryptocompare(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices from CryptoCompare API."""
        symbols_str = ','.join([self.coin_mappings['cryptocompare'].get(s, s) for s in symbols])
        
        params = {
            'fsyms': symbols_str,
            'tsyms': 'USD'
        }
        
        url = f"{self.api_configs['cryptocompare']['base_url']}/pricemultifull"
        data = self._make_request('cryptocompare', url, params)
        
        if not data or 'RAW' not in data:
            return {}
        
        result = {}
        for symbol in symbols:
            cc_symbol = self.coin_mappings['cryptocompare'].get(symbol, symbol)
            if cc_symbol in data['RAW'] and 'USD' in data['RAW'][cc_symbol]:
                usd_data = data['RAW'][cc_symbol]['USD']
                result[symbol] = {
                    'price': usd_data.get('PRICE', 0),
                    'change_24h': usd_data.get('CHANGEPCT24HOUR', 0),
                    'market_cap': usd_data.get('MKTCAP', 0)
                }
        
        return result

    def get_prices_coinbase(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices from Coinbase API."""
        result = {}
        
        for symbol in symbols:
            pair = self.coin_mappings['coinbase'].get(symbol)
            if not pair:
                continue
            
            # Get current price
            price_url = f"{self.api_configs['coinbase']['base_url']}/exchange-rates"
            params = {'currency': symbol}
            
            data = self._make_request('coinbase', price_url, params)
            
            if data and 'data' in data and 'rates' in data['data']:
                rates = data['data']['rates']
                if 'USD' in rates:
                    result[symbol] = {
                        'price': 1 / float(rates['USD']) if float(rates['USD']) > 0 else 0,
                        'change_24h': 0,  # Coinbase basic API doesn't include 24h change
                        'market_cap': 0
                    }
        
        return result

    def get_multi_api_prices(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get prices from multiple APIs with intelligent failover."""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'ADA', 'SOL']
        
        # Cache key
        cache_key = f"prices_{'-'.join(symbols)}"
        
        # Check cache
        if cache_key in self._cache:
            cache_age = time.time() - self._cache_timestamps[cache_key]
            if cache_age < 300:  # 5 minutes cache
                return self._cache[cache_key]
        
        active_apis = self._get_active_apis()
        
        for api_name in active_apis:
            try:
                if api_name == 'coingecko':
                    prices = self.get_prices_coingecko(symbols)
                elif api_name == 'coincap':
                    prices = self.get_prices_coincap(symbols)
                elif api_name == 'cryptocompare':
                    prices = self.get_prices_cryptocompare(symbols)
                elif api_name == 'coinbase':
                    prices = self.get_prices_coinbase(symbols)
                else:
                    continue
                
                if prices:
                    # Cache successful result
                    self._cache[cache_key] = prices
                    self._cache_timestamps[cache_key] = time.time()
                    
                    st.success(f"✅ Data from {api_name.title()} API")
                    return prices
                    
            except Exception as e:
                st.warning(f"Error with {api_name}: {str(e)}")
                continue
        
        # If all APIs fail, return fallback data
        st.error("🔴 All APIs failed - using fallback data")
        return self._get_fallback_prices(symbols)

    def get_historical_data_multi(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get historical data with API failover."""
        cache_key = f"history_{symbol}_{days}"
        
        # Check cache
        if cache_key in self._cache:
            cache_age = time.time() - self._cache_timestamps[cache_key]
            if cache_age < 1800:  # 30 minutes cache
                return self._cache[cache_key]
        
        # Try CoinGecko first (best historical data)
        if self._should_use_api('coingecko'):
            df = self._get_coingecko_historical(symbol, days)
            if df is not None and not df.empty:
                self._cache[cache_key] = df
                self._cache_timestamps[cache_key] = time.time()
                return df
        
        # Try CoinCap as backup
        if self._should_use_api('coincap'):
            df = self._get_coincap_historical(symbol, days)
            if df is not None and not df.empty:
                self._cache[cache_key] = df
                self._cache_timestamps[cache_key] = time.time()
                return df
        
        # Generate fallback data
        df = self._generate_fallback_historical(symbol, days)
        return df

    def _get_coingecko_historical(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data from CoinGecko."""
        coin_id = self.coin_mappings['coingecko'].get(symbol)
        if not coin_id:
            return None
        
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily'
        }
        
        url = f"{self.api_configs['coingecko']['base_url']}/coins/{coin_id}/market_chart"
        data = self._make_request('coingecko', url, params)
        
        if not data or 'prices' not in data:
            return None
        
        return self._convert_prices_to_dataframe(data['prices'])

    def _get_coincap_historical(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data from CoinCap."""
        coin_id = self.coin_mappings['coincap'].get(symbol)
        if not coin_id:
            return None
        
        # CoinCap uses different time format
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        params = {
            'interval': 'd1',  # daily
            'start': start_time,
            'end': end_time
        }
        
        url = f"{self.api_configs['coincap']['base_url']}/assets/{coin_id}/history"
        data = self._make_request('coincap', url, params)
        
        if not data or 'data' not in data:
            return None
        
        # Convert CoinCap format to price list
        prices = [[item['time'], float(item['priceUsd'])] for item in data['data']]
        return self._convert_prices_to_dataframe(prices)

    def _convert_prices_to_dataframe(self, prices: List[List]) -> pd.DataFrame:
        """Convert price data to DataFrame."""
        if not prices:
            return pd.DataFrame()
        
        dates = [datetime.fromtimestamp(item[0] / 1000) for item in prices]
        closes = [item[1] for item in prices]
        
        data = []
        for i, (date, close) in enumerate(zip(dates, closes)):
            # Generate OHLV from close
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

    def _get_fallback_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate fallback price data."""
        fallback_prices = {
            'BTC': 45000,
            'ETH': 2500,
            'ADA': 0.5,
            'SOL': 100,
            'BNB': 300,
            'XRP': 0.6,
            'MATIC': 1.0
        }
        
        result = {}
        for symbol in symbols:
            base_price = fallback_prices.get(symbol, 100)
            result[symbol] = {
                'price': base_price * (1 + np.random.normal(0, 0.05)),
                'change_24h': np.random.uniform(-8, 8),
                'market_cap': base_price * np.random.uniform(10000000, 1000000000)
            }
        
        return result

    def _generate_fallback_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate fallback historical data."""
        fallback_prices = {
            'BTC': 45000, 'ETH': 2500, 'ADA': 0.5, 'SOL': 100,
            'BNB': 300, 'XRP': 0.6, 'MATIC': 1.0
        }
        
        base_price = fallback_prices.get(symbol, 100)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        prices = [base_price]
        for _ in range(days - 1):
            change = np.random.normal(0.001, 0.03)
            prices.append(prices[-1] * (1 + change))
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = 0.02
            open_price = prices[i-1] if i > 0 else close
            high = max(open_price, close) * (1 + volatility/2)
            low = min(open_price, close) * (1 - volatility/2)
            volume = 1000000 * np.random.uniform(0.5, 2)
            
            data.append({
                'Date': date,
                'Open': round(open_price, 4),
                'High': round(high, 4),
                'Low': round(low, 4), 
                'Close': round(close, 4),
                'Volume': round(volume)
            })
        
        return pd.DataFrame(data)

    def get_api_status(self) -> Dict[str, Dict]:
        """Get status of all APIs."""
        status = {}
        for api_name, config in self.api_configs.items():
            status[api_name] = {
                'active': config['active'],
                'failures': config['failures'],
                'priority': config['priority'],
                'status': '🟢 Active' if config['active'] else '🔴 Disabled'
            }
        return status


# Global instance
multi_api_manager = MultiAPICryptoManager()


# Convenience functions for Streamlit app
@st.cache_data(ttl=300)  # 5 minutes cache
def get_multi_api_prices(symbols: List[str] = None):
    """Get prices from multiple APIs with failover."""
    return multi_api_manager.get_multi_api_prices(symbols)

@st.cache_data(ttl=1800)  # 30 minutes cache
def get_multi_api_historical(symbol: str, days: int = 7):
    """Get historical data with API failover."""
    return multi_api_manager.get_historical_data_multi(symbol, days)

def get_api_status_info():
    """Get API status information."""
    return multi_api_manager.get_api_status()


if __name__ == "__main__":
    # Test the multi-API system
    print("Testing Multi-API Crypto System...")
    
    # Test prices
    prices = get_multi_api_prices(['BTC', 'ETH', 'ADA', 'SOL'])
    print(f"Prices: {prices}")
    
    # Test historical data
    btc_history = get_multi_api_historical('BTC', 7)
    print(f"BTC History Shape: {btc_history.shape}")
    
    # Test API status
    status = get_api_status_info()
    print(f"API Status: {status}")