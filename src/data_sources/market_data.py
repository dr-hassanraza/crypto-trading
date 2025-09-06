import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
from binance.client import Client as BinanceClient
from newsapi import NewsApiClient
import ccxt
from typing import Dict, List, Any, Optional
import time
from config.config import Config

class MarketDataFetcher:
    def __init__(self):
        self.config = Config()
        self.binance_client = None
        self.news_client = None
        self.session = None
        
        # Initialize clients if API keys are available
        if self.config.BINANCE_API_KEY and self.config.BINANCE_SECRET_KEY:
            self.binance_client = BinanceClient(
                self.config.BINANCE_API_KEY, 
                self.config.BINANCE_SECRET_KEY
            )
        
        if self.config.NEWS_API_KEY:
            self.news_client = NewsApiClient(api_key=self.config.NEWS_API_KEY)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_coingecko_data(self, crypto_ids: List[str]) -> Dict[str, Any]:
        """Fetch market data from CoinGecko API."""
        base_url = "https://api.coingecko.com/api/v3"
        
        # Current prices and market data
        url = f"{base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ','.join(crypto_ids),
            'order': 'market_cap_desc',
            'per_page': len(crypto_ids),
            'page': 1,
            'sparkline': 'true',
            'price_change_percentage': '1h,24h,7d,30d'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    market_data = await response.json()
                    
                    # Get additional metrics
                    fear_greed = await self._get_fear_greed_index()
                    global_data = await self._get_global_market_data()
                    
                    return {
                        'market_data': market_data,
                        'fear_greed_index': fear_greed,
                        'global_metrics': global_data,
                        'timestamp': datetime.now(),
                        'source': 'coingecko'
                    }
                else:
                    print(f"CoinGecko API error: {response.status}")
                    return {}
        except Exception as e:
            print(f"Error fetching CoinGecko data: {e}")
            return {}
    
    async def _get_fear_greed_index(self) -> Optional[Dict]:
        """Get Fear & Greed Index."""
        try:
            url = "https://api.alternative.me/fng/"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [{}])[0]
        except:
            pass
        return None
    
    async def _get_global_market_data(self) -> Optional[Dict]:
        """Get global cryptocurrency market data."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
        except:
            pass
        return None
    
    def get_binance_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        if not self.binance_client:
            return pd.DataFrame()
        
        try:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[numeric_columns]
        
        except Exception as e:
            print(f"Error fetching Binance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, crypto_name: str, days_back: int = 7) -> Dict[str, Any]:
        """Fetch news sentiment for a cryptocurrency."""
        if not self.news_client:
            return {}
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Search for news
            articles = self.news_client.get_everything(
                q=f'"{crypto_name}" OR "{crypto_name.replace("-", " ")}" crypto cryptocurrency bitcoin',
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='popularity',
                page_size=50
            )
            
            if not articles or not articles.get('articles'):
                return {}
            
            # Analyze sentiment (simplified)
            positive_keywords = ['surge', 'rally', 'bullish', 'gains', 'up', 'rise', 'positive', 'breakthrough']
            negative_keywords = ['crash', 'dump', 'bearish', 'losses', 'down', 'fall', 'negative', 'decline']
            
            sentiment_scores = []
            article_count = len(articles['articles'])
            
            for article in articles['articles'][:20]:  # Analyze top 20 articles
                title = (article.get('title') or '').lower()
                description = (article.get('description') or '').lower()
                text = f"{title} {description}"
                
                positive_count = sum(1 for word in positive_keywords if word in text)
                negative_count = sum(1 for word in negative_keywords if word in text)
                
                if positive_count > negative_count:
                    sentiment_scores.append(1)
                elif negative_count > positive_count:
                    sentiment_scores.append(-1)
                else:
                    sentiment_scores.append(0)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            return {
                'sentiment_score': round(avg_sentiment, 3),
                'article_count': article_count,
                'analyzed_articles': len(sentiment_scores),
                'positive_ratio': sum(1 for s in sentiment_scores if s > 0) / len(sentiment_scores) if sentiment_scores else 0,
                'negative_ratio': sum(1 for s in sentiment_scores if s < 0) / len(sentiment_scores) if sentiment_scores else 0,
                'recent_headlines': [a.get('title', '')[:100] for a in articles['articles'][:5]],
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            print(f"Error fetching news sentiment for {crypto_name}: {e}")
            return {}
    
    async def get_comprehensive_data(self, crypto_id: str, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data from all sources for a single cryptocurrency."""
        
        # Fetch data from multiple sources concurrently
        tasks = [
            self.get_coingecko_data([crypto_id]),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        coingecko_data = results[0] if not isinstance(results[0], Exception) else {}
        
        # Get OHLCV data from Binance
        ohlcv_1h = self.get_binance_klines(symbol, '1h', 168)  # 1 week
        ohlcv_4h = self.get_binance_klines(symbol, '4h', 168)  # 4 weeks
        ohlcv_1d = self.get_binance_klines(symbol, '1d', 100)  # 100 days
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(crypto_id, days_back=7)
        
        # Combine all data
        comprehensive_data = {
            'crypto_id': crypto_id,
            'symbol': symbol,
            'coingecko': coingecko_data,
            'ohlcv_data': {
                '1h': ohlcv_1h,
                '4h': ohlcv_4h,
                '1d': ohlcv_1d
            },
            'news_sentiment': news_sentiment,
            'data_quality': self._assess_data_quality(coingecko_data, ohlcv_1h, news_sentiment),
            'last_updated': datetime.now()
        }
        
        return comprehensive_data
    
    def _assess_data_quality(self, coingecko: Dict, ohlcv: pd.DataFrame, news: Dict) -> Dict[str, Any]:
        """Assess the quality and completeness of fetched data."""
        quality_score = 0
        issues = []
        
        # Check CoinGecko data
        if coingecko and coingecko.get('market_data'):
            quality_score += 40
        else:
            issues.append("Missing CoinGecko market data")
        
        # Check OHLCV data
        if not ohlcv.empty and len(ohlcv) > 20:
            quality_score += 30
        else:
            issues.append("Insufficient OHLCV data")
        
        # Check news data
        if news and news.get('article_count', 0) > 0:
            quality_score += 20
        else:
            issues.append("No news data available")
        
        # Check data freshness
        if coingecko.get('timestamp'):
            data_age = (datetime.now() - coingecko['timestamp']).seconds
            if data_age < 300:  # Less than 5 minutes old
                quality_score += 10
            else:
                issues.append("Stale market data")
        
        return {
            'score': quality_score,
            'grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 70 else 'C' if quality_score >= 50 else 'D',
            'issues': issues,
            'timestamp': datetime.now()
        }
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview and sentiment."""
        try:
            # Get top cryptocurrencies
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'price_change_percentage': '1h,24h,7d'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    market_data = await response.json()
                    
                    # Calculate market metrics
                    total_market_cap = sum(coin.get('market_cap', 0) for coin in market_data if coin.get('market_cap'))
                    
                    # Count gainers/losers
                    gainers_24h = sum(1 for coin in market_data if coin.get('price_change_percentage_24h', 0) > 0)
                    losers_24h = len(market_data) - gainers_24h
                    
                    # Average changes
                    avg_change_1h = np.mean([coin.get('price_change_percentage_1h_in_currency', 0) for coin in market_data if coin.get('price_change_percentage_1h_in_currency') is not None])
                    avg_change_24h = np.mean([coin.get('price_change_percentage_24h', 0) for coin in market_data if coin.get('price_change_percentage_24h') is not None])
                    avg_change_7d = np.mean([coin.get('price_change_percentage_7d_in_currency', 0) for coin in market_data if coin.get('price_change_percentage_7d_in_currency') is not None])
                    
                    # Get Fear & Greed Index
                    fear_greed = await self._get_fear_greed_index()
                    
                    return {
                        'total_market_cap': total_market_cap,
                        'gainers_24h': gainers_24h,
                        'losers_24h': losers_24h,
                        'gainer_ratio': gainers_24h / len(market_data) if market_data else 0,
                        'avg_change_1h': round(avg_change_1h, 2),
                        'avg_change_24h': round(avg_change_24h, 2),
                        'avg_change_7d': round(avg_change_7d, 2),
                        'fear_greed_index': fear_greed,
                        'market_sentiment': self._determine_market_sentiment(avg_change_24h, gainers_24h, len(market_data), fear_greed),
                        'top_gainers': sorted([c for c in market_data if c.get('price_change_percentage_24h')], 
                                            key=lambda x: x['price_change_percentage_24h'], reverse=True)[:5],
                        'top_losers': sorted([c for c in market_data if c.get('price_change_percentage_24h')], 
                                           key=lambda x: x['price_change_percentage_24h'])[:5],
                        'timestamp': datetime.now()
                    }
            
        except Exception as e:
            print(f"Error fetching market overview: {e}")
            return {}
    
    def _determine_market_sentiment(self, avg_change: float, gainers: int, total: int, fear_greed: Optional[Dict]) -> str:
        """Determine overall market sentiment."""
        gainer_ratio = gainers / total if total > 0 else 0
        
        # Multiple factors for sentiment
        sentiment_score = 0
        
        # Price change factor
        if avg_change > 2:
            sentiment_score += 2
        elif avg_change > 0:
            sentiment_score += 1
        elif avg_change > -2:
            sentiment_score -= 1
        else:
            sentiment_score -= 2
        
        # Gainer ratio factor
        if gainer_ratio > 0.6:
            sentiment_score += 1
        elif gainer_ratio < 0.4:
            sentiment_score -= 1
        
        # Fear & Greed factor
        if fear_greed:
            fg_value = int(fear_greed.get('value', 50))
            if fg_value > 75:
                sentiment_score += 1
            elif fg_value < 25:
                sentiment_score -= 1
        
        # Determine sentiment
        if sentiment_score >= 3:
            return "VERY_BULLISH"
        elif sentiment_score >= 1:
            return "BULLISH"
        elif sentiment_score <= -3:
            return "VERY_BEARISH"
        elif sentiment_score <= -1:
            return "BEARISH"
        else:
            return "NEUTRAL"