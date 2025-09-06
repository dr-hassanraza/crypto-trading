"""
Advanced Feature Selection and Preprocessing Pipeline

High-performance feature engineering and selection system optimized for
sub-500ms processing with comprehensive market feature extraction.
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.feature_selection import (
        SelectKBest, f_regression, mutual_info_regression, 
        RFE, SelectFromModel, VarianceThreshold
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV
    import ta  # Technical analysis library
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async


@dataclass
class FeaturePipelineConfig:
    """Configuration for feature pipeline with performance constraints."""
    
    # Performance constraints
    max_processing_time_ms: int = 300  # Target sub-500ms total
    max_features: int = 100
    max_data_points: int = 50000
    enable_parallel_processing: bool = True
    
    # Feature selection parameters
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    mutual_info_k_best: int = 50
    recursive_feature_elimination: bool = False
    lasso_feature_selection: bool = True
    
    # Technical indicators
    enable_technical_indicators: bool = True
    enable_volume_indicators: bool = True
    enable_volatility_indicators: bool = True
    enable_momentum_indicators: bool = True
    
    # Market microstructure features
    enable_orderbook_features: bool = False  # Expensive to compute
    enable_flow_features: bool = True
    enable_regime_features: bool = True
    
    # Preprocessing
    scaling_method: str = 'robust'  # 'standard', 'minmax', 'robust'
    handle_missing_method: str = 'forward_fill'  # 'drop', 'interpolate', 'forward_fill'
    outlier_detection: bool = True
    outlier_threshold: float = 3.0


@dataclass
class FeatureImportance:
    """Feature importance scores from different methods."""
    feature_name: str
    variance_score: float
    mutual_info_score: float
    correlation_score: float
    lasso_importance: float
    combined_score: float
    rank: int


@dataclass
class PipelineResult:
    """Result from feature pipeline processing."""
    processed_data: np.ndarray
    feature_names: List[str]
    feature_importance: List[FeatureImportance]
    processing_time_ms: float
    original_features: int
    selected_features: int
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class TechnicalIndicatorEngine:
    """High-performance technical indicator calculation."""
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.indicator_cache = {}
    
    def calculate_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with performance optimization."""
        start_time = time.time()
        
        try:
            if not FEATURE_SELECTION_AVAILABLE:
                return ohlcv_data.copy()
            
            # Create copy to avoid modifying original data
            df = ohlcv_data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                crypto_logger.logger.warning(f"Missing OHLCV columns: {missing_cols}")
                return df
            
            # Price-based indicators (fast to compute)
            if self.config.enable_technical_indicators:
                # Moving averages (vectorized)
                df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
                df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
                df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
                
                # RSI (optimized)
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                
                # MACD (fast computation)
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = macd.macd_diff()
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df['close'])
                df['bb_high'] = bb.bollinger_hband()
                df['bb_low'] = bb.bollinger_lband()
                df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
            
            # Momentum indicators
            if self.config.enable_momentum_indicators:
                df['roc'] = ta.momentum.roc(df['close'], window=10)
                df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
                df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # Volume indicators (if volume data available)
            if self.config.enable_volume_indicators and 'volume' in df.columns:
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                df['vwap'] = ta.volume.volume_weighted_average_price(
                    df['high'], df['low'], df['close'], df['volume']
                )
                df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility indicators
            if self.config.enable_volatility_indicators:
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                df['volatility'] = df['close'].rolling(window=20).std()
                df['price_range'] = (df['high'] - df['low']) / df['close']
            
            processing_time = (time.time() - start_time) * 1000
            
            crypto_logger.logger.debug(
                f"Technical indicators calculated in {processing_time:.2f}ms"
            )
            
            return df
            
        except Exception as e:
            crypto_logger.logger.error(f"Technical indicator calculation failed: {e}")
            return ohlcv_data.copy()
    
    def calculate_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features (lightweight version)."""
        if not self.config.enable_flow_features:
            return data
        
        try:
            # Price momentum features
            data['price_momentum_1'] = data['close'].pct_change(1)
            data['price_momentum_5'] = data['close'].pct_change(5)
            data['price_momentum_15'] = data['close'].pct_change(15)
            
            # Realized volatility
            data['realized_vol_5'] = data['price_momentum_1'].rolling(5).std()
            data['realized_vol_20'] = data['price_momentum_1'].rolling(20).std()
            
            # Price efficiency measures
            data['price_efficiency'] = np.abs(data['price_momentum_1']) / (
                data['high'] - data['low']
            ).fillna(method='ffill')
            
            # Volume-price relationship
            if 'volume' in data.columns:
                data['volume_price_trend'] = data['volume'] * np.sign(data['price_momentum_1'])
                data['volume_momentum'] = data['volume'].pct_change(1)
            
            return data
            
        except Exception as e:
            crypto_logger.logger.warning(f"Microstructure features failed: {e}")
            return data


class FeatureSelector:
    """High-performance feature selection with multiple methods."""
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.feature_importance_scores = {}
        self.selected_features = []
        
    def select_features(self, X: np.ndarray, y: np.ndarray = None, 
                       feature_names: List[str] = None) -> Tuple[np.ndarray, List[str], List[FeatureImportance]]:
        """Select best features using multiple methods."""
        start_time = time.time()
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        original_features = X.shape[1]
        
        # Step 1: Remove low variance features
        X, feature_names = self._remove_low_variance_features(X, feature_names)
        
        # Step 2: Remove highly correlated features
        X, feature_names = self._remove_correlated_features(X, feature_names)
        
        # Step 3: Statistical feature selection (if target provided)
        if y is not None and len(np.unique(y)) > 1:
            X, feature_names = self._statistical_feature_selection(X, y, feature_names)
        
        # Step 4: Model-based feature selection (if target provided)
        if y is not None and self.config.lasso_feature_selection and len(np.unique(y)) > 1:
            X, feature_names = self._model_based_selection(X, y, feature_names)
        
        # Ensure we don't exceed max features
        if X.shape[1] > self.config.max_features:
            X = X[:, :self.config.max_features]
            feature_names = feature_names[:self.config.max_features]
        
        # Calculate feature importance scores
        importance_scores = self._calculate_feature_importance(X, y, feature_names)
        
        processing_time = (time.time() - start_time) * 1000
        
        crypto_logger.logger.debug(
            f"Feature selection: {original_features} → {X.shape[1]} features in {processing_time:.2f}ms"
        )
        
        return X, feature_names, importance_scores
    
    def _remove_low_variance_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove features with low variance."""
        try:
            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            X_selected = selector.fit_transform(X)
            
            selected_mask = selector.get_support()
            selected_features = [name for i, name in enumerate(feature_names) if selected_mask[i]]
            
            return X_selected, selected_features
            
        except Exception as e:
            crypto_logger.logger.warning(f"Variance threshold selection failed: {e}")
            return X, feature_names
    
    def _remove_correlated_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove highly correlated features."""
        try:
            if X.shape[1] <= 1:
                return X, feature_names
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            corr_matrix = np.abs(corr_matrix)
            
            # Find highly correlated pairs
            upper_triangle = np.triu(corr_matrix, k=1)
            high_corr_pairs = np.where(upper_triangle > self.config.correlation_threshold)
            
            # Remove features with high correlation
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                # Keep feature with higher variance
                if np.var(X[:, i]) >= np.var(X[:, j]):
                    features_to_remove.add(j)
                else:
                    features_to_remove.add(i)
            
            # Keep features
            keep_indices = [i for i in range(X.shape[1]) if i not in features_to_remove]
            X_selected = X[:, keep_indices]
            selected_features = [feature_names[i] for i in keep_indices]
            
            return X_selected, selected_features
            
        except Exception as e:
            crypto_logger.logger.warning(f"Correlation-based selection failed: {e}")
            return X, feature_names
    
    def _statistical_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select features using statistical tests."""
        try:
            # Use mutual information for non-linear relationships
            k_best = min(self.config.mutual_info_k_best, X.shape[1])
            
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=k_best
            )
            
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = [name for i, name in enumerate(feature_names) if selected_mask[i]]
            
            # Store scores for importance calculation
            scores = selector.scores_
            for i, (name, selected) in enumerate(zip(feature_names, selected_mask)):
                if selected:
                    self.feature_importance_scores[name] = {
                        'mutual_info_score': scores[i] if not np.isnan(scores[i]) else 0
                    }
            
            return X_selected, selected_features
            
        except Exception as e:
            crypto_logger.logger.warning(f"Statistical feature selection failed: {e}")
            return X, feature_names
    
    def _model_based_selection(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select features using LASSO regularization."""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000, n_jobs=1)
            lasso.fit(X, y)
            
            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > 1e-5
            
            if np.sum(selected_mask) == 0:
                # If no features selected, keep top features by coefficient magnitude
                top_indices = np.argsort(np.abs(lasso.coef_))[-20:]
                selected_mask = np.zeros_like(lasso.coef_, dtype=bool)
                selected_mask[top_indices] = True
            
            X_selected = X[:, selected_mask]
            selected_features = [name for i, name in enumerate(feature_names) if selected_mask[i]]
            
            # Store LASSO coefficients
            for i, (name, selected) in enumerate(zip(feature_names, selected_mask)):
                if selected:
                    if name not in self.feature_importance_scores:
                        self.feature_importance_scores[name] = {}
                    self.feature_importance_scores[name]['lasso_importance'] = np.abs(lasso.coef_[i])
            
            return X_selected, selected_features
            
        except Exception as e:
            crypto_logger.logger.warning(f"LASSO feature selection failed: {e}")
            return X, feature_names
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str]) -> List[FeatureImportance]:
        """Calculate comprehensive feature importance scores."""
        importance_list = []
        
        try:
            # Calculate variance scores
            variances = np.var(X, axis=0)
            
            # Calculate correlation scores (with target if available)
            correlation_scores = np.zeros(X.shape[1])
            if y is not None:
                for i in range(X.shape[1]):
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    correlation_scores[i] = np.abs(corr) if not np.isnan(corr) else 0
            
            # Combine scores
            for i, name in enumerate(feature_names):
                variance_score = variances[i]
                correlation_score = correlation_scores[i]
                
                # Get previously calculated scores
                mutual_info_score = self.feature_importance_scores.get(name, {}).get('mutual_info_score', 0)
                lasso_importance = self.feature_importance_scores.get(name, {}).get('lasso_importance', 0)
                
                # Combined score (weighted average)
                combined_score = (
                    0.25 * variance_score +
                    0.25 * correlation_score +
                    0.25 * mutual_info_score +
                    0.25 * lasso_importance
                )
                
                importance_list.append(FeatureImportance(
                    feature_name=name,
                    variance_score=variance_score,
                    mutual_info_score=mutual_info_score,
                    correlation_score=correlation_score,
                    lasso_importance=lasso_importance,
                    combined_score=combined_score,
                    rank=0  # Will be set after sorting
                ))
            
            # Sort by combined score and assign ranks
            importance_list.sort(key=lambda x: x.combined_score, reverse=True)
            for i, importance in enumerate(importance_list):
                importance.rank = i + 1
            
            return importance_list
            
        except Exception as e:
            crypto_logger.logger.warning(f"Feature importance calculation failed: {e}")
            return []


class AdvancedFeaturePipeline:
    """Main feature processing pipeline with performance optimization."""
    
    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        self.config = config or FeaturePipelineConfig()
        self.technical_engine = TechnicalIndicatorEngine(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.scaler = self._create_scaler()
        
        self.processing_history = []
        self.feature_cache = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2) if self.config.enable_parallel_processing else None
        
        crypto_logger.logger.info("🔧 Advanced Feature Pipeline initialized")
    
    def _create_scaler(self):
        """Create appropriate scaler based on configuration."""
        if self.config.scaling_method == 'standard':
            return StandardScaler()
        elif self.config.scaling_method == 'minmax':
            return MinMaxScaler()
        else:  # robust (default)
            return RobustScaler()
    
    async def process_market_data(self, ohlcv_data: pd.DataFrame, 
                                 target_column: str = None,
                                 custom_features: pd.DataFrame = None) -> PipelineResult:
        """Process market data through complete feature pipeline."""
        start_time = time.time()
        
        try:
            if not FEATURE_SELECTION_AVAILABLE:
                raise ImportError("Feature selection dependencies not available")
            
            # Validate input data
            if ohlcv_data.empty:
                return self._create_empty_result()
            
            # Limit data size for performance
            if len(ohlcv_data) > self.config.max_data_points:
                ohlcv_data = ohlcv_data.tail(self.config.max_data_points)
                crypto_logger.logger.warning(
                    f"Data truncated to {self.config.max_data_points} points for performance"
                )
            
            # Step 1: Calculate technical indicators
            feature_data = self.technical_engine.calculate_indicators(ohlcv_data)
            
            # Step 2: Add microstructure features
            feature_data = self.technical_engine.calculate_market_microstructure_features(feature_data)
            
            # Step 3: Add custom features if provided
            if custom_features is not None:
                feature_data = pd.concat([feature_data, custom_features], axis=1)
            
            # Step 4: Handle missing values
            feature_data = self._handle_missing_values(feature_data)
            
            # Step 5: Prepare data for feature selection
            numeric_columns = feature_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column from features if it exists
            if target_column and target_column in numeric_columns:
                numeric_columns.remove(target_column)
            
            X = feature_data[numeric_columns].values
            feature_names = numeric_columns
            
            # Prepare target variable
            y = None
            if target_column and target_column in feature_data.columns:
                y = feature_data[target_column].values
                # Remove NaN from target
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
            
            original_features = len(feature_names)
            
            # Step 6: Feature selection
            X_selected, selected_features, importance_scores = self.feature_selector.select_features(
                X, y, feature_names
            )
            
            # Step 7: Handle outliers
            if self.config.outlier_detection:
                X_selected = self._handle_outliers(X_selected)
            
            # Step 8: Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # Calculate performance metrics
            total_time = (time.time() - start_time) * 1000
            
            # Performance check
            performance_target_met = total_time < self.config.max_processing_time_ms
            
            if not performance_target_met:
                crypto_logger.logger.warning(
                    f"Feature pipeline exceeded target: {total_time:.2f}ms > {self.config.max_processing_time_ms}ms"
                )
            
            # Create result
            result = PipelineResult(
                processed_data=X_scaled,
                feature_names=selected_features,
                feature_importance=importance_scores,
                processing_time_ms=total_time,
                original_features=original_features,
                selected_features=len(selected_features),
                performance_metrics={
                    'performance_target_met': performance_target_met,
                    'data_points_processed': X_scaled.shape[0],
                    'missing_values_handled': feature_data.isnull().sum().sum(),
                    'outliers_detected': 0  # Would be calculated in outlier detection
                },
                metadata={
                    'scaling_method': self.config.scaling_method,
                    'target_column': target_column,
                    'technical_indicators_enabled': self.config.enable_technical_indicators
                }
            )
            
            # Store performance history
            self.processing_history.append({
                'timestamp': datetime.now(),
                'processing_time_ms': total_time,
                'data_points': X_scaled.shape[0],
                'features_original': original_features,
                'features_selected': len(selected_features),
                'performance_target_met': performance_target_met
            })
            
            # Limit history size
            if len(self.processing_history) > 100:
                self.processing_history = self.processing_history[-100:]
            
            crypto_logger.logger.info(
                f"✅ Feature pipeline: {original_features} → {len(selected_features)} features "
                f"in {total_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'feature_pipeline'})
            return self._create_empty_result()
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        if self.config.handle_missing_method == 'drop':
            return data.dropna()
        elif self.config.handle_missing_method == 'interpolate':
            return data.interpolate(method='linear')
        else:  # forward_fill
            return data.fillna(method='ffill').fillna(method='bfill')
    
    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Handle outliers using z-score method."""
        try:
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outlier_mask = z_scores > self.config.outlier_threshold
            
            # Replace outliers with median values
            for i in range(X.shape[1]):
                column_outliers = outlier_mask[:, i]
                if np.any(column_outliers):
                    median_value = np.median(X[~column_outliers, i])
                    X[column_outliers, i] = median_value
            
            return X
            
        except Exception as e:
            crypto_logger.logger.warning(f"Outlier handling failed: {e}")
            return X
    
    def _create_empty_result(self) -> PipelineResult:
        """Create empty result for error cases."""
        return PipelineResult(
            processed_data=np.array([]),
            feature_names=[],
            feature_importance=[],
            processing_time_ms=0.0,
            original_features=0,
            selected_features=0,
            performance_metrics={},
            metadata={}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        if not self.processing_history:
            return {}
        
        recent_times = [p['processing_time_ms'] for p in self.processing_history[-20:]]
        recent_performance = [p['performance_target_met'] for p in self.processing_history[-20:]]
        
        return {
            'avg_processing_time_ms': np.mean(recent_times),
            'max_processing_time_ms': np.max(recent_times),
            'min_processing_time_ms': np.min(recent_times),
            'performance_target_success_rate': np.mean(recent_performance),
            'total_pipeline_runs': len(self.processing_history),
            'avg_feature_reduction_ratio': np.mean([
                1 - (p['features_selected'] / max(p['features_original'], 1))
                for p in self.processing_history[-20:]
            ])
        }
    
    def update_config(self, new_config: FeaturePipelineConfig):
        """Update pipeline configuration."""
        self.config = new_config
        self.technical_engine = TechnicalIndicatorEngine(new_config)
        self.feature_selector = FeatureSelector(new_config)
        self.scaler = self._create_scaler()
        crypto_logger.logger.info("Feature pipeline configuration updated")


# Global feature pipeline instance
feature_pipeline = AdvancedFeaturePipeline()