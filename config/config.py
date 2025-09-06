
import os
from typing import Optional

class Config:
    """Configuration settings for Crypto Trend Analyzer."""
    
    # API Keys (set these in environment variables)
    BINANCE_API_KEY: Optional[str] = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY: Optional[str] = os.getenv('BINANCE_SECRET_KEY')
    COINBASE_API_KEY: Optional[str] = os.getenv('COINBASE_API_KEY')
    COINBASE_SECRET_KEY: Optional[str] = os.getenv('COINBASE_SECRET_KEY')
    
    # Social Media APIs
    TWITTER_API_KEY: Optional[str] = os.getenv('TWITTER_API_KEY')
    TWITTER_SECRET_KEY: Optional[str] = os.getenv('TWITTER_SECRET_KEY')
    REDDIT_CLIENT_ID: Optional[str] = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_SECRET: Optional[str] = os.getenv('REDDIT_SECRET')
    
    # OpenAI API
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    # News API
    NEWS_API_KEY: Optional[str] = os.getenv('NEWS_API_KEY')
    
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///crypto_analyzer.db')
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    
    # App Settings
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
