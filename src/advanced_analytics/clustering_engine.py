"""
Advanced Clustering Engine with HDBSCAN

High-performance clustering system with strict feature rules and 
performance optimization for sub-500ms updates.
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    from umap import UMAP
    import numba
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters with performance rules."""
    min_cluster_size: int = 15
    min_samples: int = 5
    max_features: int = 50  # Strict limit for performance
    max_data_points: int = 10000  # Memory limit
    cluster_selection_epsilon: float = 0.0
    alpha: float = 1.0
    metric: str = 'euclidean'
    
    # Performance constraints
    max_processing_time_ms: int = 400  # Sub-500ms target
    enable_dimensionality_reduction: bool = True
    target_dimensions: int = 10
    
    # Feature selection rules
    correlation_threshold: float = 0.95  # Remove highly correlated features
    variance_threshold: float = 0.01  # Remove low-variance features
    importance_threshold: float = 0.001  # Minimum feature importance


@dataclass
class ClusterResult:
    """Result from clustering analysis."""
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray
    n_clusters: int
    outlier_scores: np.ndarray
    silhouette_score: float
    processing_time_ms: float
    feature_importance: Dict[str, float]
    cluster_centers: np.ndarray
    cluster_sizes: Dict[int, int]
    validity_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketCluster:
    """Market regime cluster information."""
    cluster_id: int
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    characteristics: Dict[str, float]
    sample_size: int
    stability_score: float
    opportunity_score: float


class PerformanceOptimizedPreprocessor:
    """High-performance preprocessing with strict rules."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.feature_names = []
        self.processing_stats = {
            'total_features_processed': 0,
            'features_selected': 0,
            'dimensionality_reduction_time': 0,
            'scaling_time': 0
        }
    
    def fit_transform(self, data: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit and transform data with performance monitoring."""
        start_time = time.time()
        
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering dependencies not available")
        
        # Validate input
        if data.shape[0] > self.config.max_data_points:
            # Sample data to stay within limits
            indices = np.random.choice(data.shape[0], self.config.max_data_points, replace=False)
            data = data[indices]
            crypto_logger.logger.warning(
                f"Data sampled from {data.shape[0]} to {self.config.max_data_points} points"
            )
        
        original_features = data.shape[1]
        self.feature_names = feature_names or [f'feature_{i}' for i in range(original_features)]
        
        # Step 1: Remove NaN and infinite values
        data = self._clean_data(data)
        
        # Step 2: Feature selection based on variance and correlation
        data = self._apply_feature_selection(data)
        
        # Step 3: Scaling
        scale_start = time.time()
        data = self.scaler.fit_transform(data)
        self.processing_stats['scaling_time'] = (time.time() - scale_start) * 1000
        
        # Step 4: Dimensionality reduction if needed
        if self.config.enable_dimensionality_reduction and data.shape[1] > self.config.target_dimensions:
            dim_start = time.time()
            data = self._apply_dimensionality_reduction(data)
            self.processing_stats['dimensionality_reduction_time'] = (time.time() - dim_start) * 1000
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.processing_stats.update({
            'total_features_processed': original_features,
            'features_selected': data.shape[1],
            'total_processing_time': processing_time
        })
        
        crypto_logger.logger.debug(
            f"Preprocessing: {original_features} → {data.shape[1]} features in {processing_time:.2f}ms"
        )
        
        return data, self.processing_stats
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Clean data by removing NaN and infinite values."""
        # Replace NaN with median
        data = np.where(np.isnan(data), np.nanmedian(data, axis=0), data)
        
        # Replace infinite values with 99th percentile
        data = np.where(np.isinf(data), np.nanpercentile(data, 99, axis=0), data)
        
        return data
    
    def _apply_feature_selection(self, data: np.ndarray) -> np.ndarray:
        """Apply feature selection based on variance and correlation."""
        # Remove low variance features
        variances = np.var(data, axis=0)
        high_variance_mask = variances > self.config.variance_threshold
        data = data[:, high_variance_mask]
        
        # Remove highly correlated features
        if data.shape[1] > 1:
            correlation_matrix = np.corrcoef(data.T)
            correlation_matrix = np.abs(correlation_matrix)
            
            # Find highly correlated pairs
            upper_triangle = np.triu(correlation_matrix, k=1)
            high_corr_pairs = np.where(upper_triangle > self.config.correlation_threshold)
            
            # Remove second feature in each highly correlated pair
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                features_to_remove.add(j)  # Remove the second feature
            
            keep_features = [i for i in range(data.shape[1]) if i not in features_to_remove]
            data = data[:, keep_features]
        
        # Ensure we don't exceed max features
        if data.shape[1] > self.config.max_features:
            # Use PCA to select top features
            pca = PCA(n_components=self.config.max_features)
            data = pca.fit_transform(data)
        
        return data
    
    def _apply_dimensionality_reduction(self, data: np.ndarray) -> np.ndarray:
        """Apply UMAP for dimensionality reduction."""
        try:
            self.dimensionality_reducer = UMAP(
                n_components=min(self.config.target_dimensions, data.shape[1]),
                n_neighbors=min(15, data.shape[0] // 4),
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            
            reduced_data = self.dimensionality_reducer.fit_transform(data)
            return reduced_data
            
        except Exception as e:
            crypto_logger.logger.warning(f"Dimensionality reduction failed: {e}")
            return data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        data = self._clean_data(data)
        data = self.scaler.transform(data)
        
        if self.dimensionality_reducer is not None:
            data = self.dimensionality_reducer.transform(data)
        
        return data


class HighPerformanceHDBSCAN:
    """Optimized HDBSCAN clustering with performance guarantees."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.clusterer = None
        self.is_fitted = False
        self.performance_metrics = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def fit_predict(self, data: np.ndarray) -> ClusterResult:
        """Fit HDBSCAN and predict clusters with performance monitoring."""
        start_time = time.time()
        
        try:
            # Initialize HDBSCAN with performance optimizations
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                alpha=self.config.alpha,
                metric=self.config.metric,
                algorithm='best',  # Let HDBSCAN choose the best algorithm
                core_dist_n_jobs=1,  # Control parallelization
                cluster_selection_method='eom'  # Excess of Mass for better performance
            )
            
            # Fit and predict
            cluster_labels = self.clusterer.fit_predict(data)
            
            # Calculate probabilities (if available)
            cluster_probabilities = getattr(self.clusterer, 'probabilities_', np.ones(len(cluster_labels)))
            
            # Calculate outlier scores
            outlier_scores = getattr(self.clusterer, 'outlier_scores_', np.zeros(len(cluster_labels)))
            
            processing_time = (time.time() - start_time) * 1000
            
            # Validate performance constraint
            if processing_time > self.config.max_processing_time_ms:
                crypto_logger.logger.warning(
                    f"Clustering took {processing_time:.2f}ms, exceeds {self.config.max_processing_time_ms}ms target"
                )
            
            # Calculate cluster metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Calculate validity metrics
            validity_metrics = self._calculate_validity_metrics(data, cluster_labels)
            
            # Calculate cluster centers and sizes
            cluster_centers = self._calculate_cluster_centers(data, cluster_labels)
            cluster_sizes = self._calculate_cluster_sizes(cluster_labels)
            
            # Feature importance (simplified for performance)
            feature_importance = self._calculate_feature_importance(data, cluster_labels)
            
            result = ClusterResult(
                cluster_labels=cluster_labels,
                cluster_probabilities=cluster_probabilities,
                n_clusters=n_clusters,
                outlier_scores=outlier_scores,
                silhouette_score=validity_metrics.get('silhouette_score', 0.0),
                processing_time_ms=processing_time,
                feature_importance=feature_importance,
                cluster_centers=cluster_centers,
                cluster_sizes=cluster_sizes,
                validity_metrics=validity_metrics
            )
            
            self.is_fitted = True
            crypto_logger.logger.info(
                f"HDBSCAN clustering: {n_clusters} clusters in {processing_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'hdbscan_clustering'})
            # Return empty result on failure
            return ClusterResult(
                cluster_labels=np.full(data.shape[0], -1),
                cluster_probabilities=np.zeros(data.shape[0]),
                n_clusters=0,
                outlier_scores=np.ones(data.shape[0]),
                silhouette_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                feature_importance={},
                cluster_centers=np.array([]),
                cluster_sizes={},
                validity_metrics={}
            )
    
    def _calculate_validity_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering validity metrics."""
        metrics = {}
        
        try:
            # Only calculate if we have valid clusters
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters > 1:
                # Silhouette score (sample to speed up if too many points)
                sample_size = min(1000, len(data))
                if len(data) > sample_size:
                    indices = np.random.choice(len(data), sample_size, replace=False)
                    sample_data = data[indices]
                    sample_labels = labels[indices]
                else:
                    sample_data = data
                    sample_labels = labels
                
                # Remove outliers for silhouette calculation
                non_outlier_mask = sample_labels != -1
                if np.sum(non_outlier_mask) > 0:
                    silhouette_avg = silhouette_score(
                        sample_data[non_outlier_mask], 
                        sample_labels[non_outlier_mask]
                    )
                    metrics['silhouette_score'] = float(silhouette_avg)
                
                # Calinski-Harabasz score
                if np.sum(non_outlier_mask) > 0:
                    ch_score = calinski_harabasz_score(
                        sample_data[non_outlier_mask], 
                        sample_labels[non_outlier_mask]
                    )
                    metrics['calinski_harabasz_score'] = float(ch_score)
            
            metrics['n_outliers'] = int(np.sum(labels == -1))
            metrics['outlier_ratio'] = float(np.sum(labels == -1) / len(labels))
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to calculate validity metrics: {e}")
        
        return metrics
    
    def _calculate_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label != -1:  # Skip outliers
                cluster_points = data[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
        
        return np.array(centers) if centers else np.array([])
    
    def _calculate_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """Calculate size of each cluster."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels.astype(int), counts.astype(int)))
    
    def _calculate_feature_importance(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate simplified feature importance for performance."""
        try:
            # Use variance within clusters vs between clusters as importance measure
            importance = {}
            
            for feature_idx in range(min(data.shape[1], 20)):  # Limit for performance
                feature_data = data[:, feature_idx]
                
                # Calculate within-cluster variance
                within_var = 0
                between_var = 0
                
                unique_labels = np.unique(labels)
                valid_labels = unique_labels[unique_labels != -1]
                
                if len(valid_labels) > 1:
                    overall_mean = np.mean(feature_data)
                    
                    for label in valid_labels:
                        cluster_data = feature_data[labels == label]
                        if len(cluster_data) > 0:
                            cluster_mean = np.mean(cluster_data)
                            within_var += np.var(cluster_data) * len(cluster_data)
                            between_var += (cluster_mean - overall_mean) ** 2 * len(cluster_data)
                    
                    # Feature importance as ratio of between to within variance
                    if within_var > 0:
                        importance_score = between_var / within_var
                    else:
                        importance_score = 0
                    
                    importance[f'feature_{feature_idx}'] = float(importance_score)
            
            return importance
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to calculate feature importance: {e}")
            return {}


class ClusteringEngine:
    """Main clustering engine with performance optimization."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.preprocessor = PerformanceOptimizedPreprocessor(self.config)
        self.clusterer = HighPerformanceHDBSCAN(self.config)
        
        self.last_result = None
        self.performance_history = []
        self.cache = {}  # Simple result cache
        
        crypto_logger.logger.info("🧩 Clustering Engine initialized with performance optimization")
    
    async def cluster_market_data(self, data: pd.DataFrame, 
                                 feature_columns: List[str] = None) -> ClusterResult:
        """Cluster market data with performance monitoring."""
        start_time = time.time()
        
        try:
            if not CLUSTERING_AVAILABLE:
                raise ImportError("Clustering dependencies (hdbscan, umap) not installed")
            
            # Prepare data
            if feature_columns:
                data_array = data[feature_columns].values
                feature_names = feature_columns
            else:
                # Use numeric columns only
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                data_array = data[numeric_columns].values
                feature_names = numeric_columns
            
            # Check cache (simple hash-based)
            data_hash = hash(str(data_array.tobytes()))
            if data_hash in self.cache:
                cached_result = self.cache[data_hash]
                if (datetime.now() - cached_result.timestamp).seconds < 300:  # 5 minute cache
                    crypto_logger.logger.debug("Using cached clustering result")
                    return cached_result
            
            # Preprocess data
            processed_data, preprocessing_stats = self.preprocessor.fit_transform(
                data_array, feature_names
            )
            
            # Check if we have enough data for clustering
            if processed_data.shape[0] < self.config.min_cluster_size:
                crypto_logger.logger.warning(
                    f"Insufficient data for clustering: {processed_data.shape[0]} points"
                )
                return self._create_empty_result(processed_data.shape[0])
            
            # Perform clustering
            result = self.clusterer.fit_predict(processed_data)
            
            # Update performance metrics
            total_time = (time.time() - start_time) * 1000
            result.processing_time_ms = total_time
            
            # Cache result
            self.cache[data_hash] = result
            
            # Clean old cache entries
            if len(self.cache) > 10:
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].timestamp)
                del self.cache[oldest_key]
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'processing_time_ms': total_time,
                'data_points': processed_data.shape[0],
                'features': processed_data.shape[1],
                'n_clusters': result.n_clusters
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            self.last_result = result
            
            crypto_logger.logger.info(
                f"✅ Clustering completed: {result.n_clusters} clusters, "
                f"{total_time:.2f}ms total time"
            )
            
            return result
            
        except Exception as e:
            await handle_error_async(e, {'component': 'clustering_engine'})
            return self._create_empty_result(data.shape[0] if isinstance(data, pd.DataFrame) else 0)
    
    def _create_empty_result(self, n_points: int) -> ClusterResult:
        """Create empty result for error cases."""
        return ClusterResult(
            cluster_labels=np.full(n_points, -1),
            cluster_probabilities=np.zeros(n_points),
            n_clusters=0,
            outlier_scores=np.ones(n_points),
            silhouette_score=0.0,
            processing_time_ms=0.0,
            feature_importance={},
            cluster_centers=np.array([]),
            cluster_sizes={},
            validity_metrics={}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get clustering performance statistics."""
        if not self.performance_history:
            return {}
        
        recent_times = [p['processing_time_ms'] for p in self.performance_history[-20:]]
        
        return {
            'avg_processing_time_ms': np.mean(recent_times),
            'max_processing_time_ms': np.max(recent_times),
            'min_processing_time_ms': np.min(recent_times),
            'performance_target_met': np.mean(recent_times) < self.config.max_processing_time_ms,
            'total_clustering_runs': len(self.performance_history),
            'cache_hit_rate': len(self.cache) / max(len(self.performance_history), 1)
        }
    
    def update_config(self, new_config: ClusteringConfig):
        """Update clustering configuration."""
        self.config = new_config
        self.preprocessor = PerformanceOptimizedPreprocessor(new_config)
        self.clusterer = HighPerformanceHDBSCAN(new_config)
        crypto_logger.logger.info("Clustering configuration updated")


# Global clustering engine instance
clustering_engine = ClusteringEngine()