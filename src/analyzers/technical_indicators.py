import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volume import VolumeSMAIndicator, OnBalanceVolumeIndicator
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given dataframe."""
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Basic price indicators
        df = self._add_price_indicators(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        # Momentum indicators  
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Support/Resistance levels
        df = self._add_support_resistance(df)
        
        return df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
        
        # Exponential Moving Averages
        for period in [12, 26, 50, 200]:
            df[f'EMA_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
        
        # Price position relative to moving averages
        df['price_above_sma_20'] = (df['close'] > df['SMA_20']).astype(int)
        df['price_above_sma_50'] = (df['close'] > df['SMA_50']).astype(int)
        df['price_above_ema_12'] = (df['close'] > df['EMA_12']).astype(int)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend following indicators."""
        # MACD
        macd = MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        df['MACD_bullish'] = (df['MACD'] > df['MACD_signal']).astype(int)
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['ADX'] = adx.adx()
        df['ADX_pos'] = adx.adx_pos()
        df['ADX_neg'] = adx.adx_neg()
        df['ADX_strong_trend'] = (df['ADX'] > 25).astype(int)
        
        # Moving Average Convergence
        df['MA_convergence'] = np.where(
            (df['SMA_10'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50']), 1,
            np.where((df['SMA_10'] < df['SMA_20']) & (df['SMA_20'] < df['SMA_50']), -1, 0)
        )
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        rsi = RSIIndicator(df['close'])
        df['RSI'] = rsi.rsi()
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        df['Stoch_oversold'] = (df['Stoch_K'] < 20).astype(int)
        df['Stoch_overbought'] = (df['Stoch_K'] > 80).astype(int)
        
        # Rate of Change
        df['ROC_10'] = df['close'].pct_change(10) * 100
        df['ROC_20'] = df['close'].pct_change(20) * 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        bb = BollingerBands(df['close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volatility measures
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        df['high_volatility'] = (df['volatility_20'] > df['volatility_20'].rolling(50).mean() * 1.5).astype(int)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        # Volume SMA
        df['Volume_SMA_20'] = VolumeSMAIndicator(df['close'], df['volume']).volume_sma()
        
        # On Balance Volume
        df['OBV'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels."""
        # Pivot points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['support_1'] = 2 * df['pivot'] - df['high']
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_2'] = df['pivot'] - (df['high'] - df['low'])
        df['resistance_2'] = df['pivot'] + (df['high'] - df['low'])
        
        # Recent highs/lows
        df['recent_high_20'] = df['high'].rolling(20).max()
        df['recent_low_20'] = df['low'].rolling(20).min()
        
        return df
    
    def get_signal_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall signal strength based on multiple indicators."""
        if len(df) < 50:
            return {"strength": 0, "direction": "HOLD", "confidence": 0}
        
        latest = df.iloc[-1]
        signals = []
        
        # Trend signals
        if latest['MACD_bullish']:
            signals.append(1)
        else:
            signals.append(-1)
            
        if latest['price_above_sma_20']:
            signals.append(1)
        else:
            signals.append(-1)
            
        if latest['MA_convergence'] == 1:
            signals.append(2)  # Stronger signal
        elif latest['MA_convergence'] == -1:
            signals.append(-2)
        else:
            signals.append(0)
        
        # Momentum signals
        if latest['RSI'] < 30:
            signals.append(2)  # Strong buy signal
        elif latest['RSI'] > 70:
            signals.append(-2)  # Strong sell signal
        elif 40 <= latest['RSI'] <= 60:
            signals.append(0)  # Neutral
        else:
            signals.append(1 if latest['RSI'] < 50 else -1)
        
        # Volume confirmation
        if latest['high_volume']:
            volume_multiplier = 1.2
        else:
            volume_multiplier = 0.8
        
        # Calculate overall signal
        signal_sum = sum(signals) * volume_multiplier
        max_possible = 8 * volume_multiplier
        
        strength = signal_sum / max_possible
        confidence = min(abs(strength) * 100, 95)  # Cap at 95%
        
        if strength > 0.3:
            direction = "BUY"
        elif strength < -0.3:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        return {
            "strength": round(strength, 3),
            "direction": direction,
            "confidence": round(confidence, 1),
            "signals": signals,
            "key_levels": {
                "resistance": latest['resistance_1'],
                "support": latest['support_1'],
                "pivot": latest['pivot']
            }
        }
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect common chart patterns."""
        patterns = {}
        
        if len(df) < 20:
            return patterns
        
        recent_data = df.tail(20)
        
        # Golden Cross / Death Cross
        if len(df) > 50:
            sma_50_current = df['SMA_50'].iloc[-1]
            sma_50_prev = df['SMA_50'].iloc[-2]
            sma_20_current = df['SMA_20'].iloc[-1]
            sma_20_prev = df['SMA_20'].iloc[-2]
            
            if sma_20_prev <= sma_50_prev and sma_20_current > sma_50_current:
                patterns['golden_cross'] = True
            elif sma_20_prev >= sma_50_prev and sma_20_current < sma_50_current:
                patterns['death_cross'] = True
        
        # Bollinger Band Squeeze
        bb_width_avg = df['BB_width'].tail(20).mean()
        bb_width_current = df['BB_width'].iloc[-1]
        
        if bb_width_current < bb_width_avg * 0.7:
            patterns['squeeze'] = True
        
        # RSI Divergence (simplified)
        price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        rsi_trend = recent_data['RSI'].iloc[-1] - recent_data['RSI'].iloc[0]
        
        if price_trend > 0 and rsi_trend < 0:
            patterns['bearish_divergence'] = True
        elif price_trend < 0 and rsi_trend > 0:
            patterns['bullish_divergence'] = True
        
        return patterns