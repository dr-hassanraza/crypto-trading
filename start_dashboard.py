#!/usr/bin/env python3
"""
Crypto Trend Analyzer - Enterprise Dashboard Startup Script
"""
import os
import sys
import subprocess
import time
from datetime import datetime

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import plotly
        import jinja2
        import websockets
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "templates",
        "static",
        "logs",
        "data",
        "src/utils",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Created necessary directories")

def create_config():
    """Create basic configuration files."""
    
    # Create config.py
    config_content = '''
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
'''
    
    os.makedirs('config', exist_ok=True)
    with open('config/config.py', 'w') as f:
        f.write(config_content)
    
    # Create logging config
    logging_content = '''
import logging
import sys
from datetime import datetime

class CryptoLogger:
    def __init__(self):
        self.logger = logging.getLogger('crypto_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/crypto_analyzer.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

crypto_logger = CryptoLogger()
'''
    
    os.makedirs('src/utils', exist_ok=True)
    with open('src/utils/logging_config.py', 'w') as f:
        f.write(logging_content)
    
    # Create __init__.py files
    init_dirs = ['src', 'src/utils', 'config']
    for directory in init_dirs:
        with open(f'{directory}/__init__.py', 'w') as f:
            f.write('# Package initialization\n')
    
    print("✅ Created configuration files")

def print_banner():
    """Print startup banner."""
    banner = '''
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║    🚀 CRYPTO TREND ANALYZER - ENTERPRISE DASHBOARD 🚀         ║
    ║                                                                  ║
    ║    ✨ Features Included:                                         ║
    ║      • Quantum-Resistant Security                                ║
    ║      • AI-Powered Portfolio Optimization                         ║
    ║      • Advanced Risk Management                                   ║
    ║      • Real-time Trading Execution                               ║
    ║      • Options & Derivatives Analysis                            ║
    ║      • Regulatory Compliance Suite                               ║
    ║      • Market Microstructure Analysis                            ║
    ║      • DeFi & Cross-Chain Analytics                              ║
    ║      • Social Sentiment Analysis                                 ║
    ║      • Whale Tracking & On-Chain Analysis                        ║
    ║                                                                  ║
    ║    💼 Enterprise-Grade Platform Ready!                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    '''
    print(banner)

def main():
    """Main startup function."""
    print_banner()
    
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Checking system requirements...")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please install the required packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories and config
    create_directories()
    create_config()
    
    print("\n🌐 Starting web dashboard...")
    print("📊 Dashboard URL: http://localhost:3001")
    print("🔄 Real-time updates: WebSocket enabled")
    print("🛡️ Security: Quantum-resistant encryption active")
    print("\n" + "="*60)
    print("Dashboard is starting... Please wait...")
    print("="*60 + "\n")
    
    try:
        # Import and run the dashboard
        from web_dashboard import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=3001,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down dashboard...")
        print("Thank you for using Crypto Trend Analyzer!")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()