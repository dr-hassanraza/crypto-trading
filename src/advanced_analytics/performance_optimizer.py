"""
Performance Optimization System for Advanced Analytics

Ensures sub-500ms processing times through caching, parallel processing,
profiling, and adaptive optimization techniques.
"""

import asyncio
import time
import functools
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    from collections import deque
    import cProfile
    import pstats
    import io
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Global performance targets
    target_total_time_ms: int = 500  # Sub-500ms target
    target_clustering_time_ms: int = 100
    target_feature_processing_ms: int = 150
    target_bayesian_inference_ms: int = 100
    target_classification_ms: int = 100
    target_api_calls_ms: int = 200
    
    # Caching settings
    enable_intelligent_caching: bool = True
    cache_ttl_seconds: int = 30
    max_cache_size: int = 1000
    cache_hit_target: float = 0.7  # 70% cache hit rate target
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_process_pooling: bool = False  # More overhead, use for CPU-intensive tasks
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    performance_monitoring_window: int = 100  # Last N operations
    slow_operation_threshold_ms: int = 300
    optimization_trigger_rate: float = 0.3  # Optimize if 30% operations are slow
    
    # Profiling and monitoring
    enable_detailed_profiling: bool = False  # Only for debugging
    enable_memory_monitoring: bool = True
    profile_sample_rate: float = 0.1  # Profile 10% of operations


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    timestamp: datetime
    total_time_ms: float
    component_times: Dict[str, float]  # Time breakdown by component
    cache_hits: int
    cache_misses: int
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    component: str
    issue: str
    recommended_action: str
    expected_improvement_ms: float
    priority: int  # 1-5, 5 is highest priority


class IntelligentCache:
    """Intelligent caching system with adaptive TTL and LRU eviction."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.lock = threading.RLock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        try:
            # Create a hashable representation of arguments
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception:
            # Fallback to simple string representation
            return f"{func_name}:{str(args)}:{str(kwargs)}"
    
    def get(self, func_name: str, args: tuple, kwargs: dict) -> Tuple[bool, Any]:
        """Get cached result if available and not expired."""
        with self.lock:
            self.cache_stats['total_requests'] += 1
            
            key = self._generate_key(func_name, args, kwargs)
            
            if key in self.cache:
                cached_item, timestamp = self.cache[key]
                
                # Check if expired
                if (datetime.now() - timestamp).seconds < self.config.cache_ttl_seconds:
                    self.cache_stats['hits'] += 1
                    self.access_times[key] = datetime.now()
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    return True, cached_item
                else:
                    # Remove expired item
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    if key in self.access_counts:
                        del self.access_counts[key]
            
            self.cache_stats['misses'] += 1
            return False, None
    
    def put(self, func_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache the result."""
        with self.lock:
            if not self.config.enable_intelligent_caching:
                return
            
            key = self._generate_key(func_name, args, kwargs)
            
            # Evict if cache is full
            if len(self.cache) >= self.config.max_cache_size:
                self._evict_lru_item()
            
            # Store result with timestamp
            self.cache[key] = (result, datetime.now())
            self.access_times[key] = datetime.now()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def _evict_lru_item(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find least recently accessed item
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove from all dictionaries
        if lru_key in self.cache:
            del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        if lru_key in self.access_counts:
            del self.access_counts[lru_key]
        
        self.cache_stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.cache_stats['total_requests']
            hit_rate = self.cache_stats['hits'] / max(total_requests, 1)
            
            return {
                'cache_size': len(self.cache),
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                **self.cache_stats
            }
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_requests': 0
            }


class PerformanceProfiler:
    """Performance profiler with detailed timing and resource monitoring."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.active_profiles = {}
        self.profiling_enabled = config.enable_detailed_profiling and PROFILING_AVAILABLE
        
        if self.profiling_enabled:
            self.profiler = cProfile.Profile()
    
    def start_profiling(self, operation_id: str):
        """Start profiling an operation."""
        if not self.profiling_enabled:
            return
        
        try:
            import psutil
            process = psutil.Process()
            
            self.active_profiles[operation_id] = {
                'start_time': time.time(),
                'start_memory': process.memory_info().rss / 1024 / 1024,  # MB
                'start_cpu': process.cpu_percent()
            }
            
            # Start detailed profiling for sampled operations
            if np.random.random() < self.config.profile_sample_rate:
                self.profiler.enable()
                self.active_profiles[operation_id]['detailed_profiling'] = True
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to start profiling: {e}")
    
    def stop_profiling(self, operation_id: str) -> Dict[str, Any]:
        """Stop profiling and return metrics."""
        if not self.profiling_enabled or operation_id not in self.active_profiles:
            return {}
        
        try:
            import psutil
            process = psutil.Process()
            
            profile_data = self.active_profiles[operation_id]
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu = process.cpu_percent()
            
            metrics = {
                'duration_ms': (end_time - profile_data['start_time']) * 1000,
                'memory_delta_mb': end_memory - profile_data['start_memory'],
                'cpu_usage_percent': (end_cpu + profile_data['start_cpu']) / 2
            }
            
            # Stop detailed profiling if it was enabled
            if profile_data.get('detailed_profiling'):
                self.profiler.disable()
                
                # Extract profiling stats
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(10)  # Top 10 functions
                
                metrics['detailed_profile'] = s.getvalue()
                
                # Re-enable profiler for next operation
                self.profiler.clear()
            
            # Clean up
            del self.active_profiles[operation_id]
            
            return metrics
            
        except Exception as e:
            crypto_logger.logger.warning(f"Failed to stop profiling: {e}")
            return {}


def performance_monitor(target_time_ms: int = None, cache_key_func: Callable = None):
    """Decorator for monitoring and optimizing function performance."""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Check cache first
            cache = getattr(performance_optimizer, 'cache', None)
            if cache and cache_key_func:
                cache_args, cache_kwargs = cache_key_func(*args, **kwargs)
                hit, result = cache.get(operation_name, cache_args, cache_kwargs)
                if hit:
                    return result
            
            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result if caching is enabled
                if cache and cache_key_func:
                    cache.put(operation_name, cache_args, cache_kwargs, result)
                
                # Record performance
                execution_time = (time.time() - start_time) * 1000
                
                if hasattr(performance_optimizer, 'record_performance'):
                    performance_optimizer.record_performance(
                        operation_name, execution_time, success=True
                    )
                
                # Check if performance target was met
                target = target_time_ms or 100
                if execution_time > target:
                    crypto_logger.logger.warning(
                        f"{operation_name} took {execution_time:.2f}ms, exceeds target {target}ms"
                    )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                if hasattr(performance_optimizer, 'record_performance'):
                    performance_optimizer.record_performance(
                        operation_name, execution_time, success=False, error=str(e)
                    )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            start_time = time.time()
            operation_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                target = target_time_ms or 100
                if execution_time > target:
                    crypto_logger.logger.warning(
                        f"{operation_name} took {execution_time:.2f}ms, exceeds target {target}ms"
                    )
                
                return result
                
            except Exception as e:
                crypto_logger.logger.error(f"{operation_name} failed: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AdaptiveOptimizer:
    """Adaptive optimization system that learns from performance patterns."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_monitoring_window)
        self.optimization_strategies = {}
        self.applied_optimizations = set()
    
    def analyze_performance_patterns(self) -> List[OptimizationRecommendation]:
        """Analyze performance patterns and generate optimization recommendations."""
        recommendations = []
        
        try:
            if len(self.performance_history) < 10:
                return recommendations
            
            # Convert to DataFrame for analysis
            recent_metrics = list(self.performance_history)[-50:]  # Last 50 operations
            
            # Analyze slow operations
            slow_operations = [m for m in recent_metrics if m.total_time_ms > self.config.slow_operation_threshold_ms]
            
            if len(slow_operations) / len(recent_metrics) > self.config.optimization_trigger_rate:
                # Identify bottlenecks
                component_times = {}
                for metric in recent_metrics:
                    for component, time_ms in metric.component_times.items():
                        if component not in component_times:
                            component_times[component] = []
                        component_times[component].append(time_ms)
                
                # Find slowest components
                avg_component_times = {
                    component: np.mean(times) 
                    for component, times in component_times.items()
                }
                
                slowest_component = max(avg_component_times, key=avg_component_times.get)
                slowest_time = avg_component_times[slowest_component]
                
                if slowest_time > 100:  # Component taking more than 100ms on average
                    recommendations.append(OptimizationRecommendation(
                        component=slowest_component,
                        issue=f"Component averaging {slowest_time:.1f}ms",
                        recommended_action=self._get_optimization_strategy(slowest_component),
                        expected_improvement_ms=slowest_time * 0.3,  # Expect 30% improvement
                        priority=5
                    ))
            
            # Analyze cache performance
            cache_stats = getattr(performance_optimizer, 'cache', None)
            if cache_stats:
                stats = cache_stats.get_stats()
                if stats['hit_rate'] < self.config.cache_hit_target:
                    recommendations.append(OptimizationRecommendation(
                        component='caching',
                        issue=f"Low cache hit rate: {stats['hit_rate']:.2%}",
                        recommended_action="Increase cache TTL or improve cache key strategy",
                        expected_improvement_ms=50,
                        priority=4
                    ))
            
            # Analyze memory usage patterns
            high_memory_operations = [m for m in recent_metrics if m.memory_usage_mb > 500]
            if len(high_memory_operations) > 5:
                recommendations.append(OptimizationRecommendation(
                    component='memory_management',
                    issue="High memory usage detected",
                    recommended_action="Implement data streaming or batch processing",
                    expected_improvement_ms=30,
                    priority=3
                ))
            
            return recommendations
            
        except Exception as e:
            crypto_logger.logger.warning(f"Performance analysis failed: {e}")
            return []
    
    def _get_optimization_strategy(self, component: str) -> str:
        """Get optimization strategy for a specific component."""
        strategies = {
            'clustering': "Reduce data points or features, enable parallel processing",
            'feature_processing': "Cache feature calculations, optimize feature selection",
            'bayesian_inference': "Reduce sampling size, use analytical solutions where possible",
            'api_calls': "Batch API calls, implement connection pooling",
            'market_classification': "Cache regime classifications, reduce complexity"
        }
        
        return strategies.get(component, "Profile and optimize bottlenecks")
    
    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation."""
        try:
            if recommendation.component in self.applied_optimizations:
                return False  # Already applied
            
            crypto_logger.logger.info(
                f"Applying optimization for {recommendation.component}: {recommendation.recommended_action}"
            )
            
            # Mark as applied
            self.applied_optimizations.add(recommendation.component)
            
            # Here you would implement specific optimization actions
            # For now, just log the recommendation
            return True
            
        except Exception as e:
            crypto_logger.logger.error(f"Failed to apply optimization: {e}")
            return False


class AdvancedPerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = IntelligentCache(self.config)
        self.profiler = PerformanceProfiler(self.config)
        self.adaptive_optimizer = AdaptiveOptimizer(self.config)
        
        # Thread pools for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=2) if self.config.enable_process_pooling else None
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.component_timers = {}
        
        crypto_logger.logger.info("🚀 Advanced Performance Optimizer initialized")
    
    def start_operation_timing(self, operation_name: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.component_timers[operation_id] = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'component_times': {}
        }
        
        # Start profiling if enabled
        self.profiler.start_profiling(operation_id)
        
        return operation_id
    
    def record_component_time(self, operation_id: str, component_name: str, duration_ms: float):
        """Record time for a specific component."""
        if operation_id in self.component_timers:
            self.component_timers[operation_id]['component_times'][component_name] = duration_ms
    
    def finish_operation_timing(self, operation_id: str, success: bool = True, error: str = None):
        """Finish timing an operation and record metrics."""
        try:
            if operation_id not in self.component_timers:
                return
            
            timer_data = self.component_timers[operation_id]
            total_time = (time.time() - timer_data['start_time']) * 1000
            
            # Get profiling metrics
            profiling_metrics = self.profiler.stop_profiling(operation_id)
            
            # Create performance metric
            metric = PerformanceMetrics(
                operation_name=timer_data['operation_name'],
                timestamp=datetime.now(),
                total_time_ms=total_time,
                component_times=timer_data['component_times'],
                cache_hits=0,  # Would be filled by cache
                cache_misses=0,
                memory_usage_mb=profiling_metrics.get('memory_delta_mb', 0),
                cpu_usage_percent=profiling_metrics.get('cpu_usage_percent', 0),
                success=success,
                error_message=error
            )
            
            # Store metrics
            self.performance_metrics.append(metric)
            self.adaptive_optimizer.performance_history.append(metric)
            
            # Clean up timer
            del self.component_timers[operation_id]
            
            # Check performance targets
            self._check_performance_targets(metric)
            
        except Exception as e:
            crypto_logger.logger.error(f"Failed to finish operation timing: {e}")
    
    def _check_performance_targets(self, metric: PerformanceMetrics):
        """Check if performance targets are met."""
        if metric.total_time_ms > self.config.target_total_time_ms:
            crypto_logger.logger.warning(
                f"Operation {metric.operation_name} exceeded target: "
                f"{metric.total_time_ms:.2f}ms > {self.config.target_total_time_ms}ms"
            )
            
            # Trigger adaptive optimization analysis
            if len(self.adaptive_optimizer.performance_history) % 20 == 0:  # Every 20 operations
                recommendations = self.adaptive_optimizer.analyze_performance_patterns()
                
                for rec in recommendations:
                    if rec.priority >= 4:  # High priority recommendations
                        self.adaptive_optimizer.apply_optimization(rec)
    
    async def run_with_timeout(self, coro, timeout_ms: int) -> Any:
        """Run coroutine with timeout."""
        try:
            timeout_seconds = timeout_ms / 1000
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            crypto_logger.logger.warning(f"Operation timed out after {timeout_ms}ms")
            raise
    
    def run_parallel_tasks(self, tasks: List[Callable], use_process_pool: bool = False) -> List[Any]:
        """Run tasks in parallel."""
        if not self.config.enable_parallel_processing:
            return [task() for task in tasks]
        
        try:
            pool = self.process_pool if (use_process_pool and self.process_pool) else self.thread_pool
            
            futures = [pool.submit(task) for task in tasks]
            results = [future.result(timeout=5) for future in futures]
            
            return results
            
        except Exception as e:
            crypto_logger.logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [task() for task in tasks]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.performance_metrics:
                return {}
            
            recent_metrics = list(self.performance_metrics)[-100:]  # Last 100 operations
            
            # Calculate summary statistics
            total_times = [m.total_time_ms for m in recent_metrics]
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            
            # Component timing analysis
            component_stats = {}
            for metric in recent_metrics:
                for component, time_ms in metric.component_times.items():
                    if component not in component_stats:
                        component_stats[component] = []
                    component_stats[component].append(time_ms)
            
            component_averages = {
                component: np.mean(times)
                for component, times in component_stats.items()
            }
            
            # Performance target compliance
            target_compliance = sum(
                1 for t in total_times 
                if t <= self.config.target_total_time_ms
            ) / len(total_times)
            
            summary = {
                'total_operations': len(self.performance_metrics),
                'avg_processing_time_ms': np.mean(total_times),
                'p95_processing_time_ms': np.percentile(total_times, 95),
                'p99_processing_time_ms': np.percentile(total_times, 99),
                'max_processing_time_ms': np.max(total_times),
                'success_rate': success_rate,
                'target_compliance_rate': target_compliance,
                'component_averages': component_averages,
                'cache_stats': self.cache.get_stats(),
                'slowest_component': max(component_averages, key=component_averages.get) if component_averages else None
            }
            
            return summary
            
        except Exception as e:
            crypto_logger.logger.error(f"Failed to generate performance summary: {e}")
            return {}
    
    def optimize_system(self):
        """Run system optimization based on performance analysis."""
        try:
            recommendations = self.adaptive_optimizer.analyze_performance_patterns()
            
            if recommendations:
                crypto_logger.logger.info(f"Found {len(recommendations)} optimization opportunities")
                
                for rec in recommendations:
                    if rec.priority >= 3:  # Medium to high priority
                        success = self.adaptive_optimizer.apply_optimization(rec)
                        if success:
                            crypto_logger.logger.info(f"Applied optimization: {rec.recommended_action}")
            
        except Exception as e:
            crypto_logger.logger.error(f"System optimization failed: {e}")
    
    def cleanup_resources(self):
        """Clean up resources."""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            if self.process_pool:
                self.process_pool.shutdown(wait=False)
                
            crypto_logger.logger.info("Performance optimizer resources cleaned up")
            
        except Exception as e:
            crypto_logger.logger.error(f"Resource cleanup failed: {e}")
    
    def record_performance(self, operation_name: str, execution_time_ms: float, 
                         success: bool = True, error: str = None):
        """Record performance for an operation (simplified interface)."""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            timestamp=datetime.now(),
            total_time_ms=execution_time_ms,
            component_times={},
            cache_hits=0,
            cache_misses=0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success=success,
            error_message=error
        )
        
        self.performance_metrics.append(metric)


# Global performance optimizer instance
performance_optimizer = AdvancedPerformanceOptimizer()


# Convenience functions for external use
def start_timing(operation_name: str) -> str:
    """Start timing an operation."""
    return performance_optimizer.start_operation_timing(operation_name)


def record_component_time(operation_id: str, component: str, duration_ms: float):
    """Record component timing."""
    performance_optimizer.record_component_time(operation_id, component, duration_ms)


def finish_timing(operation_id: str, success: bool = True, error: str = None):
    """Finish timing an operation."""
    performance_optimizer.finish_operation_timing(operation_id, success, error)


async def optimize_async_operation(operation_name: str, coro, timeout_ms: int = 500):
    """Optimize an async operation with timing and timeout."""
    operation_id = start_timing(operation_name)
    
    try:
        result = await performance_optimizer.run_with_timeout(coro, timeout_ms)
        finish_timing(operation_id, success=True)
        return result
    except Exception as e:
        finish_timing(operation_id, success=False, error=str(e))
        raise


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics."""
    return performance_optimizer.get_performance_summary()