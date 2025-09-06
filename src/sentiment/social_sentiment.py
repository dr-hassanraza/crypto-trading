import asyncio
import aiohttp
import tweepy
import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
import logging

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class SentimentData:
    platform: str
    symbol: str
    sentiment_score: float  # -1 to 1
    volume: int  # Number of mentions
    engagement: int  # Likes, retweets, upvotes
    trending_score: float
    fear_greed_indicator: float
    whale_sentiment: Optional[float]
    influencer_sentiment: float
    timestamp: datetime
    sample_posts: List[str]

@dataclass
class TrendingTopic:
    keyword: str
    platform: str
    mention_count: int
    growth_rate: float  # % increase in mentions
    sentiment_score: float
    related_cryptos: List[str]
    time_period: str
    first_seen: datetime

@dataclass
class InfluencerSignal:
    username: str
    platform: str
    follower_count: int
    post_content: str
    sentiment_score: float
    engagement: int
    influence_score: float  # Weighted by followers and engagement
    mentioned_cryptos: List[str]
    timestamp: datetime

class SocialSentimentAnalyzer:
    """Advanced social media sentiment analysis for crypto markets."""
    
    def __init__(self):
        self.config = Config()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc', '$btc', '#bitcoin'],
            'ethereum': ['ethereum', 'eth', '$eth', '#ethereum'],
            'solana': ['solana', 'sol', '$sol', '#solana'],
            'cardano': ['cardano', 'ada', '$ada', '#cardano'],
            'polkadot': ['polkadot', 'dot', '$dot', '#polkadot'],
            'chainlink': ['chainlink', 'link', '$link', '#chainlink'],
            'avalanche': ['avalanche', 'avax', '$avax', '#avalanche'],
            'polygon': ['polygon', 'matic', '$matic', '#polygon']
        }
        
        # Initialize APIs
        self.twitter_api = None
        self.reddit_api = None
        self._initialize_apis()
        
        # Influencer tracking
        self.crypto_influencers = {
            'twitter': [
                'elonmusk', 'michael_saylor', 'coindesk', 'cointelegraph',
                'APompliano', 'satoshi_nakamo', 'VitalikButerin', 'CZ_binance',
                'naval', 'tylerwinklevoss', 'cameron', 'DocumentingBTC'
            ],
            'reddit': ['cryptocurrency', 'bitcoin', 'ethereum', 'defi']
        }
        
    def _initialize_apis(self):
        """Initialize social media API connections."""
        try:
            # Twitter API v2
            if hasattr(self.config, 'TWITTER_BEARER_TOKEN') and self.config.TWITTER_BEARER_TOKEN:
                self.twitter_api = tweepy.Client(bearer_token=self.config.TWITTER_BEARER_TOKEN)
                crypto_logger.logger.info("✓ Twitter API initialized")
            
            # Reddit API
            if (hasattr(self.config, 'REDDIT_CLIENT_ID') and 
                hasattr(self.config, 'REDDIT_CLIENT_SECRET')):
                self.reddit_api = praw.Reddit(
                    client_id=self.config.REDDIT_CLIENT_ID,
                    client_secret=self.config.REDDIT_CLIENT_SECRET,
                    user_agent='CryptoSentimentBot/1.0'
                )
                crypto_logger.logger.info("✓ Reddit API initialized")
        
        except Exception as e:
            crypto_logger.logger.warning(f"Social API initialization warning: {e}")
    
    async def analyze_crypto_sentiment(self, symbols: List[str], 
                                     timeframe_hours: int = 24) -> Dict[str, SentimentData]:
        """Analyze sentiment for given crypto symbols across platforms."""
        crypto_logger.logger.info(f"Analyzing sentiment for {symbols} over {timeframe_hours}h")
        
        sentiment_data = {}
        
        # Analyze each symbol
        for symbol in symbols:
            crypto_name = symbol.lower().replace('usdt', '').replace('usd', '')
            
            # Gather sentiment from all platforms
            twitter_sentiment = await self._analyze_twitter_sentiment(crypto_name, timeframe_hours)
            reddit_sentiment = await self._analyze_reddit_sentiment(crypto_name, timeframe_hours)
            news_sentiment = await self._analyze_news_sentiment(crypto_name, timeframe_hours)
            
            # Aggregate sentiment data
            combined_sentiment = self._aggregate_sentiment_data(
                symbol, twitter_sentiment, reddit_sentiment, news_sentiment
            )
            
            sentiment_data[symbol] = combined_sentiment
        
        return sentiment_data
    
    async def _analyze_twitter_sentiment(self, crypto_name: str, 
                                       timeframe_hours: int) -> Dict[str, Any]:
        """Analyze Twitter sentiment for a cryptocurrency."""
        if not self.twitter_api:
            return {}
        
        try:
            keywords = self.crypto_keywords.get(crypto_name, [crypto_name])
            
            # Search tweets
            tweets_data = []
            for keyword in keywords[:2]:  # Limit to avoid rate limits
                try:
                    tweets = tweepy.Paginator(
                        self.twitter_api.search_recent_tweets,
                        query=f"{keyword} -is:retweet lang:en",
                        max_results=100,
                        tweet_fields=['created_at', 'public_metrics', 'author_id']
                    ).flatten(limit=200)
                    
                    for tweet in tweets:
                        # Check if tweet is within timeframe
                        tweet_age = (datetime.now(tweet.created_at.tzinfo) - tweet.created_at).total_seconds() / 3600
                        if tweet_age <= timeframe_hours:
                            # Analyze sentiment
                            sentiment = self.sentiment_analyzer.polarity_scores(tweet.text)
                            
                            tweets_data.append({
                                'text': tweet.text,
                                'sentiment': sentiment['compound'],
                                'engagement': tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count'],
                                'created_at': tweet.created_at,
                                'author_id': tweet.author_id
                            })
                
                except Exception as e:
                    crypto_logger.logger.debug(f"Error fetching tweets for {keyword}: {e}")
            
            if not tweets_data:
                return {}
            
            # Calculate aggregate metrics
            sentiments = [t['sentiment'] for t in tweets_data]
            engagements = [t['engagement'] for t in tweets_data]
            
            return {
                'platform': 'twitter',
                'volume': len(tweets_data),
                'avg_sentiment': np.mean(sentiments),
                'weighted_sentiment': np.average(sentiments, weights=engagements) if sum(engagements) > 0 else np.mean(sentiments),
                'total_engagement': sum(engagements),
                'sentiment_variance': np.var(sentiments),
                'sample_tweets': [t['text'][:100] for t in tweets_data[:5]]
            }
        
        except Exception as e:
            crypto_logger.logger.error(f"Twitter sentiment analysis error: {e}")
            return {}
    
    async def _analyze_reddit_sentiment(self, crypto_name: str, 
                                      timeframe_hours: int) -> Dict[str, Any]:
        """Analyze Reddit sentiment for a cryptocurrency."""
        if not self.reddit_api:
            return {}
        
        try:
            # Search relevant subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'defi', 'altcoins']
            posts_data = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    # Search recent posts
                    for post in subreddit.search(crypto_name, time_filter='day', limit=50):
                        post_age = (datetime.now().timestamp() - post.created_utc) / 3600
                        
                        if post_age <= timeframe_hours:
                            # Analyze post title and content
                            text = f"{post.title} {post.selftext}"
                            sentiment = self.sentiment_analyzer.polarity_scores(text)
                            
                            posts_data.append({
                                'title': post.title,
                                'text': text[:200],
                                'sentiment': sentiment['compound'],
                                'score': post.score,
                                'comments': post.num_comments,
                                'subreddit': subreddit_name,
                                'created_at': datetime.fromtimestamp(post.created_utc)
                            })
                
                except Exception as e:
                    crypto_logger.logger.debug(f"Error fetching from r/{subreddit_name}: {e}")
            
            if not posts_data:
                return {}
            
            # Calculate metrics
            sentiments = [p['sentiment'] for p in posts_data]
            scores = [max(p['score'], 1) for p in posts_data]  # Avoid zero weights
            
            return {
                'platform': 'reddit',
                'volume': len(posts_data),
                'avg_sentiment': np.mean(sentiments),
                'weighted_sentiment': np.average(sentiments, weights=scores),
                'total_engagement': sum(p['score'] + p['comments'] for p in posts_data),
                'sentiment_variance': np.var(sentiments),
                'sample_posts': [p['title'] for p in posts_data[:5]]
            }
        
        except Exception as e:
            crypto_logger.logger.error(f"Reddit sentiment analysis error: {e}")
            return {}
    
    async def _analyze_news_sentiment(self, crypto_name: str, 
                                    timeframe_hours: int) -> Dict[str, Any]:
        """Analyze news sentiment using web scraping and APIs."""
        try:
            # Use existing news API integration from market_data.py
            from src.data_sources.market_data import MarketDataFetcher
            
            async with MarketDataFetcher() as fetcher:
                news_data = fetcher.get_news_sentiment(crypto_name, days_back=1)
            
            if news_data:
                return {
                    'platform': 'news',
                    'volume': news_data.get('article_count', 0),
                    'avg_sentiment': news_data.get('sentiment_score', 0),
                    'weighted_sentiment': news_data.get('sentiment_score', 0),
                    'total_engagement': news_data.get('article_count', 0) * 100,  # Proxy metric
                    'sample_posts': news_data.get('recent_headlines', [])
                }
            
            return {}
        
        except Exception as e:
            crypto_logger.logger.debug(f"News sentiment analysis error: {e}")
            return {}
    
    def _aggregate_sentiment_data(self, symbol: str, twitter_data: Dict, 
                                reddit_data: Dict, news_data: Dict) -> SentimentData:
        """Aggregate sentiment data from all platforms."""
        
        # Weights for different platforms
        platform_weights = {
            'twitter': 0.4,
            'reddit': 0.35,
            'news': 0.25
        }
        
        # Collect valid sentiment scores and weights
        sentiment_scores = []
        weights = []
        total_volume = 0
        total_engagement = 0
        sample_posts = []
        
        for platform_data, weight in [(twitter_data, platform_weights['twitter']),
                                     (reddit_data, platform_weights['reddit']),
                                     (news_data, platform_weights['news'])]:
            if platform_data and 'weighted_sentiment' in platform_data:
                sentiment_scores.append(platform_data['weighted_sentiment'])
                weights.append(weight)
                total_volume += platform_data.get('volume', 0)
                total_engagement += platform_data.get('total_engagement', 0)
                sample_posts.extend(platform_data.get('sample_posts', [])[:2])
        
        # Calculate weighted average sentiment
        if sentiment_scores and weights:
            overall_sentiment = np.average(sentiment_scores, weights=weights)
        else:
            overall_sentiment = 0
        
        # Calculate trending score
        trending_score = self._calculate_trending_score(total_volume, total_engagement)
        
        # Calculate fear/greed indicator
        fear_greed = self._calculate_fear_greed_indicator(overall_sentiment, total_volume)
        
        return SentimentData(
            platform='aggregated',
            symbol=symbol,
            sentiment_score=overall_sentiment,
            volume=total_volume,
            engagement=total_engagement,
            trending_score=trending_score,
            fear_greed_indicator=fear_greed,
            whale_sentiment=None,  # Would require whale wallet analysis
            influencer_sentiment=self._analyze_influencer_sentiment(symbol),
            timestamp=datetime.now(),
            sample_posts=sample_posts[:5]
        )
    
    def _calculate_trending_score(self, volume: int, engagement: int) -> float:
        """Calculate trending score based on volume and engagement."""
        # Normalize scores (these thresholds would be calibrated with real data)
        volume_score = min(volume / 1000, 1.0)  # Normalize to 0-1
        engagement_score = min(engagement / 10000, 1.0)  # Normalize to 0-1
        
        return (volume_score * 0.6 + engagement_score * 0.4) * 100
    
    def _calculate_fear_greed_indicator(self, sentiment: float, volume: int) -> float:
        """Calculate custom fear/greed indicator."""
        # Base fear/greed on sentiment and volume
        sentiment_component = (sentiment + 1) / 2 * 100  # Convert -1,1 to 0,100
        
        # Adjust for volume (high volume amplifies sentiment)
        volume_multiplier = min(1 + (volume / 1000), 2.0)  # Max 2x multiplier
        
        fear_greed = sentiment_component * volume_multiplier
        return max(0, min(100, fear_greed))  # Clamp to 0-100
    
    def _analyze_influencer_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from crypto influencers (placeholder)."""
        # This would analyze posts from known crypto influencers
        # For now, return neutral sentiment
        return 0.0
    
    async def detect_trending_topics(self, timeframe_hours: int = 4) -> List[TrendingTopic]:
        """Detect trending crypto-related topics."""
        trending_topics = []
        
        try:
            # Twitter trending topics
            if self.twitter_api:
                twitter_trends = await self._get_twitter_trends()
                trending_topics.extend(twitter_trends)
            
            # Reddit trending posts
            if self.reddit_api:
                reddit_trends = await self._get_reddit_trends()
                trending_topics.extend(reddit_trends)
        
        except Exception as e:
            crypto_logger.logger.error(f"Error detecting trending topics: {e}")
        
        # Sort by growth rate and return top trends
        trending_topics.sort(key=lambda x: x.growth_rate, reverse=True)
        return trending_topics[:10]
    
    async def _get_twitter_trends(self) -> List[TrendingTopic]:
        """Get trending topics from Twitter."""
        trends = []
        
        try:
            # Get trending hashtags related to crypto
            crypto_hashtags = ['#bitcoin', '#crypto', '#defi', '#nft', '#web3']
            
            for hashtag in crypto_hashtags:
                tweets = tweepy.Paginator(
                    self.twitter_api.search_recent_tweets,
                    query=f"{hashtag} -is:retweet",
                    max_results=100
                ).flatten(limit=200)
                
                tweet_count = sum(1 for _ in tweets)
                
                if tweet_count > 50:  # Significant volume
                    trends.append(TrendingTopic(
                        keyword=hashtag,
                        platform='twitter',
                        mention_count=tweet_count,
                        growth_rate=50.0,  # Would calculate actual growth
                        sentiment_score=0.0,  # Would analyze sentiment
                        related_cryptos=[],  # Would extract mentioned cryptos
                        time_period='4h',
                        first_seen=datetime.now()
                    ))
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error getting Twitter trends: {e}")
        
        return trends
    
    async def _get_reddit_trends(self) -> List[TrendingTopic]:
        """Get trending topics from Reddit."""
        trends = []
        
        try:
            subreddit = self.reddit_api.subreddit('cryptocurrency')
            hot_posts = list(subreddit.hot(limit=25))
            
            for post in hot_posts:
                if post.score > 500:  # High engagement threshold
                    trends.append(TrendingTopic(
                        keyword=post.title[:50],
                        platform='reddit',
                        mention_count=post.num_comments,
                        growth_rate=post.score / 100,  # Proxy for growth
                        sentiment_score=0.0,  # Would analyze sentiment
                        related_cryptos=[],  # Would extract mentioned cryptos
                        time_period='4h',
                        first_seen=datetime.fromtimestamp(post.created_utc)
                    ))
        
        except Exception as e:
            crypto_logger.logger.debug(f"Error getting Reddit trends: {e}")
        
        return trends
    
    async def track_influencer_signals(self) -> List[InfluencerSignal]:
        """Track signals from crypto influencers."""
        signals = []
        
        if not self.twitter_api:
            return signals
        
        try:
            for username in self.crypto_influencers['twitter'][:5]:  # Limit to avoid rate limits
                try:
                    user = self.twitter_api.get_user(username=username)
                    
                    tweets = self.twitter_api.get_users_tweets(
                        user.id, max_results=10, 
                        tweet_fields=['created_at', 'public_metrics']
                    )
                    
                    if tweets.data:
                        for tweet in tweets.data:
                            # Analyze for crypto mentions and sentiment
                            mentioned_cryptos = self._extract_crypto_mentions(tweet.text)
                            
                            if mentioned_cryptos:
                                sentiment = self.sentiment_analyzer.polarity_scores(tweet.text)
                                
                                # Calculate influence score
                                influence_score = self._calculate_influence_score(
                                    user.public_metrics['followers_count'],
                                    tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count']
                                )
                                
                                signals.append(InfluencerSignal(
                                    username=username,
                                    platform='twitter',
                                    follower_count=user.public_metrics['followers_count'],
                                    post_content=tweet.text[:200],
                                    sentiment_score=sentiment['compound'],
                                    engagement=tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count'],
                                    influence_score=influence_score,
                                    mentioned_cryptos=mentioned_cryptos,
                                    timestamp=tweet.created_at
                                ))
                
                except Exception as e:
                    crypto_logger.logger.debug(f"Error tracking {username}: {e}")
        
        except Exception as e:
            crypto_logger.logger.error(f"Error tracking influencer signals: {e}")
        
        # Sort by influence score
        signals.sort(key=lambda x: x.influence_score, reverse=True)
        return signals[:20]
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text."""
        mentioned_cryptos = []
        text_lower = text.lower()
        
        for crypto, keywords in self.crypto_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_cryptos.append(crypto)
        
        return mentioned_cryptos
    
    def _calculate_influence_score(self, followers: int, engagement: int) -> float:
        """Calculate influence score based on followers and engagement."""
        # Logarithmic scaling for followers to avoid dominance by mega-accounts
        follower_score = np.log10(max(followers, 1)) / 7  # Normalize roughly to 0-1
        engagement_score = min(engagement / 10000, 1.0)  # Normalize to 0-1
        
        return (follower_score * 0.7 + engagement_score * 0.3) * 100
    
    def generate_sentiment_report(self, sentiment_data: Dict[str, SentimentData]) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis report."""
        
        if not sentiment_data:
            return {}
        
        # Calculate overall market sentiment
        all_sentiments = [data.sentiment_score for data in sentiment_data.values()]
        all_volumes = [data.volume for data in sentiment_data.values()]
        
        overall_sentiment = np.average(all_sentiments, weights=all_volumes) if sum(all_volumes) > 0 else np.mean(all_sentiments)
        
        # Find most mentioned coins
        most_mentioned = sorted(sentiment_data.items(), key=lambda x: x[1].volume, reverse=True)
        
        # Find most bullish/bearish
        most_bullish = max(sentiment_data.items(), key=lambda x: x[1].sentiment_score)
        most_bearish = min(sentiment_data.items(), key=lambda x: x[1].sentiment_score)
        
        return {
            'overall_market_sentiment': {
                'score': overall_sentiment,
                'classification': self._classify_sentiment(overall_sentiment),
                'confidence': min(sum(all_volumes) / 1000, 1.0) * 100
            },
            'individual_assets': {
                symbol: {
                    'sentiment_score': data.sentiment_score,
                    'classification': self._classify_sentiment(data.sentiment_score),
                    'volume': data.volume,
                    'trending_score': data.trending_score,
                    'fear_greed': data.fear_greed_indicator
                } for symbol, data in sentiment_data.items()
            },
            'highlights': {
                'most_mentioned': {
                    'symbol': most_mentioned[0][0],
                    'volume': most_mentioned[0][1].volume
                } if most_mentioned else None,
                'most_bullish': {
                    'symbol': most_bullish[0],
                    'sentiment': most_bullish[1].sentiment_score
                },
                'most_bearish': {
                    'symbol': most_bearish[0],
                    'sentiment': most_bearish[1].sentiment_score
                }
            },
            'sample_content': {
                symbol: data.sample_posts[:3] 
                for symbol, data in list(sentiment_data.items())[:3]
            },
            'methodology': {
                'platforms_analyzed': ['twitter', 'reddit', 'news'],
                'timeframe_hours': 24,
                'total_data_points': sum(data.volume for data in sentiment_data.values()),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories."""
        if score >= 0.3:
            return 'VERY_BULLISH'
        elif score >= 0.1:
            return 'BULLISH'
        elif score >= -0.1:
            return 'NEUTRAL'
        elif score >= -0.3:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'

# Global sentiment analyzer instance
social_sentiment_analyzer = SocialSentimentAnalyzer()