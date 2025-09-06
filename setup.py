#!/usr/bin/env python3
"""
Setup script for Crypto Trend Analyzer AI Agent
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_env_file():
    """Create .env file from template."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ .env file created. Please edit it with your API keys.")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️  .env.example not found. Creating minimal .env file...")
        with open(env_file, 'w') as f:
            f.write("""# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Binance API (Optional - for real-time data)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# News API (Optional - for sentiment analysis)  
NEWS_API_KEY=your_news_api_key

# Email Configuration (Optional - for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
TO_EMAIL=recipient@gmail.com
""")
        print("✅ Minimal .env file created. Please edit it with your API keys.")
        return True

def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'data', 'backups']
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version compatible: {sys.version}")
    return True

def install_requirements():
    """Install Python requirements."""
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  Virtual environment not detected")
        response = input("Do you want to continue installing to system Python? (y/N): ")
        if response.lower() != 'y':
            print("💡 Recommendation: Create a virtual environment first:")
            print("   python -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("   python setup.py")
            return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def run_quick_test():
    """Run a quick test to verify installation."""
    print("🧪 Running quick test...")
    
    try:
        # Test imports
        sys.path.append('src')
        from config.config import Config
        from src.utils.logging_config import crypto_logger
        from src.analyzers.technical_indicators import TechnicalAnalyzer
        
        print("✅ Core imports successful")
        
        # Test basic functionality
        config = Config()
        analyzer = TechnicalAnalyzer()
        
        print("✅ Basic initialization successful")
        print("🎉 Installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing missing dependencies")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Crypto Trend Analyzer AI Agent - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Run test
    if not run_quick_test():
        print("⚠️  Setup completed with warnings")
    else:
        print("✅ Setup completed successfully!")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run the system:")
    print("   python main.py all       # Run everything")
    print("   python main.py dashboard # Web dashboard only") 
    print("   python main.py analyze   # Analysis engine only")
    print("   python main.py backtest  # Run backtesting")
    print("3. Open dashboard: http://localhost:8000")
    print("\n💡 For help: python main.py --help")
    print("📚 Documentation: README_ADVANCED.md")

if __name__ == "__main__":
    main()