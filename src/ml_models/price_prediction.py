import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger

class AdvancedPricePredictionModel:
    """Advanced ML models for cryptocurrency price prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_scaler = MinMaxScaler()
        self.lookback_window = 60  # 60 time periods lookback
        self.prediction_horizons = [1, 6, 24, 168]  # 1h, 6h, 1d, 1w ahead
        
        # Model configurations
        self.model_configs = {
            'lstm': {'epochs': 100, 'batch_size': 32},
            'cnn_lstm': {'epochs': 100, 'batch_size': 32},
            'transformer': {'epochs': 100, 'batch_size': 32},
            'ensemble': {'n_estimators': 1000}
        }
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced features for ML models."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Technical indicators (from your existing technical analyzer)
        from src.analyzers.technical_indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df = analyzer.calculate_all_indicators(df)
        
        # Advanced features
        df = self._add_fractal_features(df)
        df = self._add_fourier_features(df)
        df = self._add_wavelet_features(df)
        df = self._add_regime_features(df)
        df = self._add_microstructure_features(df)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.dropna()
    
    def _add_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fractal dimension and Hurst exponent features."""
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent."""
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        def fractal_dimension(ts, max_lag=20):
            """Calculate fractal dimension."""
            hurst = hurst_exponent(ts, max_lag)
            return 2 - hurst
        
        window_size = 50
        df['hurst_exponent'] = df['close'].rolling(window_size).apply(
            lambda x: hurst_exponent(x.values) if len(x) == window_size else np.nan
        )
        df['fractal_dimension'] = df['close'].rolling(window_size).apply(
            lambda x: fractal_dimension(x.values) if len(x) == window_size else np.nan
        )
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform features."""
        prices = df['close'].values
        
        # Apply FFT and extract dominant frequencies
        fft = np.fft.fft(prices[-100:] if len(prices) > 100 else prices)
        freqs = np.fft.fftfreq(len(fft))
        
        # Get top 5 dominant frequencies
        dominant_freqs_idx = np.argsort(np.abs(fft))[-5:]
        
        for i, idx in enumerate(dominant_freqs_idx):
            df[f'fourier_real_{i}'] = np.real(fft[idx])
            df[f'fourier_imag_{i}'] = np.imag(fft[idx])
            df[f'fourier_freq_{i}'] = freqs[idx]
        
        return df
    
    def _add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet transform features."""
        try:
            import pywt
            
            prices = df['close'].values
            
            # Continuous wavelet transform
            scales = np.arange(1, 32)
            coefficients, frequencies = pywt.cwt(prices, scales, 'mexh')
            
            # Extract features from wavelet coefficients
            df['wavelet_energy'] = np.sum(coefficients**2, axis=0)
            df['wavelet_entropy'] = -np.sum(coefficients**2 * np.log(coefficients**2 + 1e-10), axis=0)
            df['wavelet_variance'] = np.var(coefficients, axis=0)
            
        except ImportError:
            # Fallback if pywt not available
            df['wavelet_energy'] = df['close'].rolling(20).var()
            df['wavelet_entropy'] = df['close'].rolling(20).apply(lambda x: -np.sum(x * np.log(x + 1e-10)))
            df['wavelet_variance'] = df['close'].rolling(20).var()
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features using Hidden Markov Models."""
        try:
            from hmmlearn import hmm
            
            # Prepare features for regime detection
            features = df[['returns', 'volatility']].dropna().values
            
            if len(features) > 50:
                # Fit HMM with 3 regimes (bull, bear, sideways)
                model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
                model.fit(features)
                
                # Predict regimes
                regimes = model.predict(features)
                
                # Map back to dataframe
                regime_series = pd.Series(index=df.dropna().index[-len(regimes):], data=regimes)
                df['market_regime'] = regime_series.reindex(df.index, method='ffill')
                
                # One-hot encode regimes
                for i in range(3):
                    df[f'regime_{i}'] = (df['market_regime'] == i).astype(int)
            else:
                # Fallback simple regime detection
                df['market_regime'] = 0
                for i in range(3):
                    df[f'regime_{i}'] = 0
        
        except ImportError:
            # Simple regime detection based on volatility and returns
            df['high_vol_regime'] = (df['volatility'] > df['volatility'].rolling(50).quantile(0.8)).astype(int)
            df['bull_regime'] = (df['returns'].rolling(20).mean() > 0).astype(int)
            df['bear_regime'] = (df['returns'].rolling(20).mean() < -0.01).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Bid-ask spread proxy
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['oc_gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Volume-price interaction
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['price_volume_ratio'] = df['close'] / (df['volume'] + 1)
        
        # Amihud illiquidity measure
        df['amihud_illiquidity'] = abs(df['returns']) / (df['volume'] + 1)
        
        # Kyle's lambda (price impact)
        df['price_impact'] = df['returns'] / np.sqrt(df['volume'] + 1)
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i-lookback:i])
            y.append(target[i+horizon-1])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int], output_dim: int) -> Model:
        """Build advanced LSTM model with attention."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(output_dim, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int], output_dim: int) -> Model:
        """Build CNN-LSTM hybrid model."""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(50, activation='relu'),
            Dense(output_dim, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', metrics=['mae'])
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int, int], output_dim: int) -> Model:
        """Build transformer-based model."""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = tf.keras.layers.LayerNormalization()(inputs + attention_output)
        
        # Feed Forward
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[1])(ffn_output)
        
        # Add & Norm
        ffn_output = tf.keras.layers.LayerNormalization()(attention_output + ffn_output)
        
        # Global pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        outputs = Dense(output_dim, activation='linear')(pooled)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', metrics=['mae'])
        return model
    
    def build_ensemble_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Build ensemble of traditional ML models."""
        models = {
            'rf': RandomForestRegressor(
                n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000, max_depth=6, learning_rate=0.01, 
                random_state=42, n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000, max_depth=6, learning_rate=0.01, 
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=1000, max_depth=6, learning_rate=0.01, 
                random_state=42
            )
        }
        
        return models
    
    def train_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train all ML models for price prediction."""
        crypto_logger.logger.info(f"Training ML models for {symbol}")
        
        # Prepare features
        featured_data = self.prepare_features(data)
        
        # Select features (remove non-numeric and target columns)
        feature_columns = featured_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['open', 'high', 'low', 'close']]
        
        X = featured_data[feature_columns].values
        y = featured_data['close'].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            crypto_logger.logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Scale targets
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        trained_models = {}
        
        # Train LSTM for each prediction horizon
        for horizon in self.prediction_horizons:
            crypto_logger.logger.info(f"Training LSTM for {horizon}h horizon")
            
            # Create sequences
            X_seq_train, y_seq_train = self.create_sequences(
                X_train, y_train, self.lookback_window, horizon
            )
            X_seq_test, y_seq_test = self.create_sequences(
                X_test, y_test, self.lookback_window, horizon
            )
            
            if len(X_seq_train) > 0:
                # Build and train LSTM
                lstm_model = self.build_lstm_model(
                    (self.lookback_window, X.shape[1]), 1
                )
                
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                history = lstm_model.fit(
                    X_seq_train, y_seq_train,
                    epochs=self.model_configs['lstm']['epochs'],
                    batch_size=self.model_configs['lstm']['batch_size'],
                    validation_data=(X_seq_test, y_seq_test),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate model
                train_pred = lstm_model.predict(X_seq_train)
                test_pred = lstm_model.predict(X_seq_test)
                
                train_mse = mean_squared_error(y_seq_train, train_pred)
                test_mse = mean_squared_error(y_seq_test, test_pred)
                
                trained_models[f'lstm_{horizon}h'] = {
                    'model': lstm_model,
                    'scaler': scaler,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'horizon': horizon,
                    'type': 'lstm'
                }
                
                crypto_logger.logger.info(f"LSTM {horizon}h - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        # Train ensemble models (for 1h prediction)
        crypto_logger.logger.info("Training ensemble models")
        ensemble_models = self.build_ensemble_model(X_train)
        
        for name, model in ensemble_models.items():
            try:
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                trained_models[f'ensemble_{name}'] = {
                    'model': model,
                    'scaler': scaler,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'horizon': 1,
                    'type': 'ensemble'
                }
                
                crypto_logger.logger.info(f"Ensemble {name} - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
                
            except Exception as e:
                crypto_logger.logger.error(f"Error training {name}: {e}")
        
        # Store models and metadata
        self.models[symbol] = trained_models
        self.scalers[symbol] = scaler
        self.feature_columns = feature_columns
        
        crypto_logger.logger.info(f"Training completed for {symbol}. {len(trained_models)} models trained.")
        
        return trained_models
    
    def predict_prices(self, symbol: str, current_data: pd.DataFrame, 
                      horizons: Optional[List[int]] = None) -> Dict[str, Any]:
        """Generate price predictions for multiple horizons."""
        if symbol not in self.models:
            crypto_logger.logger.warning(f"No trained models found for {symbol}")
            return {}
        
        if horizons is None:
            horizons = self.prediction_horizons
        
        # Prepare features
        featured_data = self.prepare_features(current_data)
        
        if len(featured_data) < self.lookback_window:
            crypto_logger.logger.warning(f"Insufficient data for prediction: {len(featured_data)}")
            return {}
        
        X = featured_data[self.feature_columns].values
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X)
        
        # Scale features
        scaler = self.scalers[symbol]
        X_scaled = scaler.transform(X)
        
        predictions = {}
        
        # Generate predictions for each horizon
        for horizon in horizons:
            horizon_predictions = {}
            
            # LSTM predictions
            lstm_key = f'lstm_{horizon}h'
            if lstm_key in self.models[symbol]:
                try:
                    model_info = self.models[symbol][lstm_key]
                    lstm_model = model_info['model']
                    
                    # Prepare sequence for prediction
                    if len(X_scaled) >= self.lookback_window:
                        X_seq = X_scaled[-self.lookback_window:].reshape(1, self.lookback_window, -1)
                        pred_scaled = lstm_model.predict(X_seq, verbose=0)
                        pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                        
                        horizon_predictions['lstm'] = {
                            'price': float(pred),
                            'confidence': self._calculate_prediction_confidence(model_info),
                            'model_performance': {
                                'train_mse': model_info['train_mse'],
                                'test_mse': model_info['test_mse']
                            }
                        }
                except Exception as e:
                    crypto_logger.logger.error(f"LSTM prediction error for {symbol}: {e}")
            
            # Ensemble predictions (only for 1h)
            if horizon == 1:
                ensemble_preds = []
                ensemble_weights = []
                
                for model_name in ['ensemble_rf', 'ensemble_xgb', 'ensemble_lgb', 'ensemble_gbm']:
                    if model_name in self.models[symbol]:
                        try:
                            model_info = self.models[symbol][model_name]
                            model = model_info['model']
                            
                            pred_scaled = model.predict(X_scaled[-1:])
                            pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                            
                            # Weight by inverse test MSE
                            weight = 1 / (model_info['test_mse'] + 1e-10)
                            ensemble_preds.append(pred * weight)
                            ensemble_weights.append(weight)
                            
                            horizon_predictions[model_name.replace('ensemble_', '')] = {
                                'price': float(pred),
                                'confidence': self._calculate_prediction_confidence(model_info)
                            }
                        except Exception as e:
                            crypto_logger.logger.error(f"Ensemble prediction error for {model_name}: {e}")
                
                # Weighted ensemble prediction
                if ensemble_preds:
                    weighted_pred = sum(ensemble_preds) / sum(ensemble_weights)
                    avg_confidence = np.mean([self._calculate_prediction_confidence(
                        self.models[symbol][f'ensemble_{name}']
                    ) for name in ['rf', 'xgb', 'lgb', 'gbm'] 
                    if f'ensemble_{name}' in self.models[symbol]])
                    
                    horizon_predictions['ensemble'] = {
                        'price': float(weighted_pred),
                        'confidence': avg_confidence
                    }
            
            predictions[f'{horizon}h'] = horizon_predictions
        
        # Add metadata
        current_price = featured_data['close'].iloc[-1]
        predictions['metadata'] = {
            'current_price': float(current_price),
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'data_points_used': len(featured_data)
        }
        
        return predictions
    
    def _calculate_prediction_confidence(self, model_info: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on model performance."""
        train_mse = model_info.get('train_mse', 1.0)
        test_mse = model_info.get('test_mse', 1.0)
        
        # Lower MSE = higher confidence
        # Penalize overfitting (large gap between train and test MSE)
        overfitting_penalty = max(0, (test_mse - train_mse) / train_mse)
        base_confidence = 1 / (1 + test_mse)
        
        confidence = base_confidence * (1 - overfitting_penalty * 0.5)
        return min(max(confidence * 100, 0), 95)  # 0-95% confidence
    
    def get_feature_importance(self, symbol: str) -> Dict[str, Any]:
        """Get feature importance for interpretability."""
        if symbol not in self.models:
            return {}
        
        importance_data = {}
        
        for model_name, model_info in self.models[symbol].items():
            if 'ensemble' in model_name:
                model = model_info['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = dict(zip(self.feature_columns, importances))
                    
                    # Get top 10 most important features
                    top_features = sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
                    
                    importance_data[model_name] = {
                        'top_features': top_features,
                        'all_features': feature_importance
                    }
        
        return importance_data
    
    def save_models(self, symbol: str, filepath: str):
        """Save trained models to disk."""
        if symbol not in self.models:
            return False
        
        model_data = {
            'models': {},
            'scaler': self.scalers[symbol],
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'lookback_window': self.lookback_window,
            'prediction_horizons': self.prediction_horizons
        }
        
        # Save ensemble models
        for name, info in self.models[symbol].items():
            if 'ensemble' in name:
                model_data['models'][name] = info
        
        # Save ensemble models and metadata
        joblib.dump(model_data, f"{filepath}_{symbol}_ensemble.pkl")
        
        # Save deep learning models separately
        for name, info in self.models[symbol].items():
            if 'lstm' in name:
                info['model'].save(f"{filepath}_{symbol}_{name}.h5")
        
        crypto_logger.logger.info(f"Models saved for {symbol}")
        return True
    
    def load_models(self, symbol: str, filepath: str):
        """Load trained models from disk."""
        try:
            # Load ensemble models
            ensemble_data = joblib.load(f"{filepath}_{symbol}_ensemble.pkl")
            
            self.scalers[symbol] = ensemble_data['scaler']
            self.target_scaler = ensemble_data['target_scaler']
            self.feature_columns = ensemble_data['feature_columns']
            self.lookback_window = ensemble_data['lookback_window']
            self.prediction_horizons = ensemble_data['prediction_horizons']
            
            self.models[symbol] = ensemble_data['models']
            
            # Load deep learning models
            for horizon in self.prediction_horizons:
                lstm_path = f"{filepath}_{symbol}_lstm_{horizon}h.h5"
                try:
                    lstm_model = tf.keras.models.load_model(lstm_path)
                    self.models[symbol][f'lstm_{horizon}h'] = {
                        'model': lstm_model,
                        'scaler': self.scalers[symbol],
                        'horizon': horizon,
                        'type': 'lstm'
                    }
                except:
                    pass
            
            crypto_logger.logger.info(f"Models loaded for {symbol}")
            return True
            
        except Exception as e:
            crypto_logger.logger.error(f"Error loading models for {symbol}: {e}")
            return False
    
    def get_model_performance_report(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive model performance report."""
        if symbol not in self.models:
            return {}
        
        report = {
            'symbol': symbol,
            'models': {},
            'summary': {
                'total_models': len(self.models[symbol]),
                'best_model': None,
                'best_performance': float('inf')
            }
        }
        
        for model_name, model_info in self.models[symbol].items():
            performance = {
                'train_mse': model_info.get('train_mse', 0),
                'test_mse': model_info.get('test_mse', 0),
                'horizon': model_info.get('horizon', 1),
                'type': model_info.get('type', 'unknown'),
                'confidence': self._calculate_prediction_confidence(model_info)
            }
            
            report['models'][model_name] = performance
            
            # Track best model
            if performance['test_mse'] < report['summary']['best_performance']:
                report['summary']['best_performance'] = performance['test_mse']
                report['summary']['best_model'] = model_name
        
        return report

# Global ML model instance
ml_predictor = AdvancedPricePredictionModel()