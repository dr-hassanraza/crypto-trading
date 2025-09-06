"""
Enhanced Control & Monitoring Center for Crypto Trend Analyzer

Centralized monitoring, control, and oversight system that provides
real-time visibility and manual override capabilities for the entire system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import numpy as np

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import error_handler, ErrorSeverity, ErrorCategory
from src.core.validation_models import validation_engine, ValidationResult
from src.core.black_box_processing import black_box_processor
from src.core.trading_controls import trading_control_system, TradingWindowStatus


class SystemHealth(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class AlertPriority(Enum):
    """Alert priority levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemAlert:
    """System alert information."""
    id: str
    timestamp: datetime
    priority: AlertPriority
    category: str
    title: str
    message: str
    component: str
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    trades_per_hour: int
    api_calls_per_minute: int
    error_rate: float
    latency_ms: float
    portfolio_value: float
    total_positions: int
    health_score: float


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.metrics_history = deque(maxlen=history_size)
        self.alert_callbacks: List[Callable] = []
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'latency_ms': {'warning': 1000, 'critical': 5000}
        }
    
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Application metrics (simulated/estimated)
            active_connections = len(getattr(trading_control_system, 'active_connections', []))
            
            # Get error statistics
            error_stats = error_handler.get_error_statistics()
            error_rate = error_stats.get('total_errors_last_hour', 0) / 60.0  # errors per minute
            
            # Trading metrics (would be pulled from actual systems)
            trades_per_hour = self._calculate_trades_per_hour()
            api_calls_per_minute = self._estimate_api_calls()
            latency_ms = self._measure_latency()
            portfolio_value = self._get_portfolio_value()
            total_positions = self._get_total_positions()
            
            # Calculate health score
            health_score = self._calculate_health_score(
                cpu_usage, memory_usage, error_rate, latency_ms
            )
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_connections=active_connections,
                trades_per_hour=trades_per_hour,
                api_calls_per_minute=api_calls_per_minute,
                error_rate=error_rate,
                latency_ms=latency_ms,
                portfolio_value=portfolio_value,
                total_positions=total_positions,
                health_score=health_score
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check thresholds and generate alerts
            await self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            crypto_logger.logger.error(f"Failed to collect metrics: {e}")
            # Return basic metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, active_connections=0,
                trades_per_hour=0, api_calls_per_minute=0, error_rate=1.0,
                latency_ms=9999, portfolio_value=0, total_positions=0,
                health_score=0.0
            )
    
    def _calculate_trades_per_hour(self) -> int:
        """Calculate trades per hour from recent metrics."""
        if len(self.metrics_history) < 2:
            return 0
        
        # Simple calculation based on recent data
        recent_hours = min(len(self.metrics_history), 60)  # Last hour of data
        return sum(1 for _ in range(recent_hours))  # Placeholder
    
    def _estimate_api_calls(self) -> int:
        """Estimate API calls per minute."""
        # This would integrate with actual API call tracking
        return np.random.randint(10, 50)  # Simulated
    
    def _measure_latency(self) -> float:
        """Measure system latency."""
        # This would measure actual API response times
        return np.random.uniform(50, 200)  # Simulated latency in ms
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # This would integrate with portfolio manager
        return 100000.0  # Simulated
    
    def _get_total_positions(self) -> int:
        """Get total number of positions."""
        # This would integrate with portfolio manager
        return np.random.randint(0, 10)  # Simulated
    
    def _calculate_health_score(self, cpu: float, memory: float, 
                               error_rate: float, latency: float) -> float:
        """Calculate overall system health score (0-1)."""
        # Normalize metrics to 0-1 scale (1 = best)
        cpu_score = max(0, (100 - cpu) / 100)
        memory_score = max(0, (100 - memory) / 100)
        error_score = max(0, 1 - (error_rate * 10))  # Assume 0.1 error rate = 0 score
        latency_score = max(0, 1 - (latency / 5000))  # 5s latency = 0 score
        
        # Weighted average
        weights = [0.2, 0.2, 0.3, 0.3]  # CPU, Memory, Error, Latency
        scores = [cpu_score, memory_score, error_score, latency_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # CPU usage check
        if metrics.cpu_usage >= self.thresholds['cpu_usage']['critical']:
            alerts.append(SystemAlert(
                id=f"cpu_critical_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                priority=AlertPriority.CRITICAL,
                category="performance",
                title="Critical CPU Usage",
                message=f"CPU usage at {metrics.cpu_usage:.1f}%",
                component="system_monitor"
            ))
        elif metrics.cpu_usage >= self.thresholds['cpu_usage']['warning']:
            alerts.append(SystemAlert(
                id=f"cpu_warning_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                priority=AlertPriority.WARNING,
                category="performance",
                title="High CPU Usage",
                message=f"CPU usage at {metrics.cpu_usage:.1f}%",
                component="system_monitor"
            ))
        
        # Memory usage check
        if metrics.memory_usage >= self.thresholds['memory_usage']['critical']:
            alerts.append(SystemAlert(
                id=f"memory_critical_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                priority=AlertPriority.CRITICAL,
                category="performance",
                title="Critical Memory Usage",
                message=f"Memory usage at {metrics.memory_usage:.1f}%",
                component="system_monitor"
            ))
        
        # Error rate check
        if metrics.error_rate >= self.thresholds['error_rate']['critical']:
            alerts.append(SystemAlert(
                id=f"error_critical_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                priority=AlertPriority.CRITICAL,
                category="errors",
                title="High Error Rate",
                message=f"Error rate: {metrics.error_rate:.3f} errors/min",
                component="system_monitor"
            ))
        
        # Send alerts to callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    crypto_logger.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            "time_period_hours": hours,
            "data_points": len(recent_metrics),
            "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
            "max_cpu_usage": np.max([m.cpu_usage for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "max_memory_usage": np.max([m.memory_usage for m in recent_metrics]),
            "avg_error_rate": np.mean([m.error_rate for m in recent_metrics]),
            "avg_latency": np.mean([m.latency_ms for m in recent_metrics]),
            "avg_health_score": np.mean([m.health_score for m in recent_metrics]),
            "min_health_score": np.min([m.health_score for m in recent_metrics])
        }


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.active_alerts: List[SystemAlert] = []
        self.alert_history = deque(maxlen=max_alerts)
        self.notification_callbacks: List[Callable] = []
        
    async def create_alert(self, alert: SystemAlert):
        """Create a new system alert."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Log the alert
        if alert.priority == AlertPriority.CRITICAL:
            crypto_logger.logger.critical(f"🚨 {alert.title}: {alert.message}")
        elif alert.priority == AlertPriority.HIGH:
            crypto_logger.logger.error(f"⚠️ {alert.title}: {alert.message}")
        elif alert.priority == AlertPriority.WARNING:
            crypto_logger.logger.warning(f"⚠️ {alert.title}: {alert.message}")
        else:
            crypto_logger.logger.info(f"ℹ️ {alert.title}: {alert.message}")
        
        # Notify callbacks
        for callback in self.notification_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                crypto_logger.logger.error(f"Alert notification failed: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                crypto_logger.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                crypto_logger.logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self, priority: Optional[AlertPriority] = None) -> List[SystemAlert]:
        """Get active alerts, optionally filtered by priority."""
        alerts = [a for a in self.active_alerts if not a.resolved]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active = [a for a in self.active_alerts if not a.resolved]
        
        stats = {
            "total_active": len(active),
            "unacknowledged": len([a for a in active if not a.acknowledged]),
            "by_priority": {},
            "by_category": {},
            "recent_24h": 0
        }
        
        # Count by priority and category
        for alert in active:
            priority = alert.priority.value
            category = alert.category
            
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        
        # Count recent alerts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        stats["recent_24h"] = len([
            a for a in self.alert_history 
            if a.timestamp >= cutoff
        ])
        
        return stats
    
    def add_notification_callback(self, callback: Callable):
        """Add notification callback."""
        self.notification_callbacks.append(callback)


class ControlOverrideManager:
    """Manages manual overrides and control operations."""
    
    def __init__(self):
        self.active_overrides: Dict[str, Dict[str, Any]] = {}
        self.override_history: List[Dict[str, Any]] = []
    
    async def set_trading_override(self, override_type: str, value: Any, 
                                  duration_minutes: int = 60, reason: str = ""):
        """Set trading system override."""
        override_id = f"{override_type}_{int(datetime.now().timestamp())}"
        expires_at = datetime.now() + timedelta(minutes=duration_minutes)
        
        override_data = {
            'id': override_id,
            'type': override_type,
            'value': value,
            'reason': reason,
            'set_at': datetime.now(),
            'expires_at': expires_at,
            'active': True
        }
        
        self.active_overrides[override_id] = override_data
        self.override_history.append(override_data)
        
        # Apply the override to the trading system
        if override_type == "trading_status":
            trading_control_system.window_manager.set_manual_override(
                value, duration_minutes=duration_minutes
            )
        elif override_type == "circuit_breaker_reset":
            trading_control_system.circuit_breaker_manager.manually_reset_circuit_breaker(value)
        
        crypto_logger.logger.warning(
            f"Manual override set: {override_type}={value} for {duration_minutes}min. "
            f"Reason: {reason}"
        )
        
        return override_id
    
    async def emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        override_id = await self.set_trading_override(
            "emergency_stop", True, duration_minutes=1440, reason=reason
        )
        
        trading_control_system.emergency_shutdown(reason)
        
        crypto_logger.logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
        
        return override_id
    
    def clear_override(self, override_id: str) -> bool:
        """Clear a specific override."""
        if override_id in self.active_overrides:
            self.active_overrides[override_id]['active'] = False
            
            # Clear from trading system if needed
            override_data = self.active_overrides[override_id]
            if override_data['type'] == "trading_status":
                trading_control_system.window_manager.clear_manual_override()
            
            crypto_logger.logger.info(f"Override cleared: {override_id}")
            return True
        
        return False
    
    def get_active_overrides(self) -> List[Dict[str, Any]]:
        """Get all active overrides."""
        now = datetime.now()
        active = []
        
        for override_id, data in self.active_overrides.items():
            if data['active'] and data['expires_at'] > now:
                active.append(data)
            elif data['expires_at'] <= now:
                # Auto-expire override
                data['active'] = False
        
        return sorted(active, key=lambda x: x['set_at'], reverse=True)


class ControlMonitoringCenter:
    """Main control and monitoring center."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.override_manager = ControlOverrideManager()
        
        self.system_status = SystemHealth.HEALTHY
        self.monitoring_active = False
        self._monitoring_task = None
        self._status_callbacks: List[Callable] = []
        
        # Setup alert callbacks
        self.performance_monitor.add_alert_callback(self.alert_manager.create_alert)
        self.alert_manager.add_notification_callback(self._handle_critical_alerts)
    
    async def initialize(self):
        """Initialize the control monitoring center."""
        try:
            crypto_logger.logger.info("🎛️ Initializing Control & Monitoring Center...")
            
            # Initialize subsystems
            await trading_control_system.initialize()
            
            # Start monitoring
            await self.start_monitoring()
            
            crypto_logger.logger.info("✅ Control & Monitoring Center initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Failed to initialize Control Center: {e}")
            raise
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        crypto_logger.logger.info("📊 System monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        crypto_logger.logger.info("📊 System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = await self.performance_monitor.collect_metrics()
                
                # Update trading control system
                await trading_control_system.update_system_metrics({
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'error_rate': metrics.error_rate,
                    'latency_ms': metrics.latency_ms,
                    'health_score': metrics.health_score
                })
                
                # Assess overall system health
                await self._assess_system_health(metrics)
                
                # Clean up old data periodically
                await self._periodic_cleanup()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                crypto_logger.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _assess_system_health(self, metrics: SystemMetrics):
        """Assess overall system health and update status."""
        old_status = self.system_status
        
        # Determine health based on metrics and active issues
        critical_alerts = self.alert_manager.get_active_alerts(AlertPriority.CRITICAL)
        high_alerts = self.alert_manager.get_active_alerts(AlertPriority.HIGH)
        circuit_breakers_open = trading_control_system.circuit_breaker_manager.is_circuit_breaker_open()
        
        if critical_alerts or circuit_breakers_open:
            self.system_status = SystemHealth.CRITICAL
        elif metrics.health_score < 0.3:
            self.system_status = SystemHealth.CRITICAL
        elif high_alerts or metrics.health_score < 0.5:
            self.system_status = SystemHealth.WARNING
        elif metrics.health_score < 0.7:
            self.system_status = SystemHealth.DEGRADED
        else:
            self.system_status = SystemHealth.HEALTHY
        
        # Notify if status changed
        if old_status != self.system_status:
            await self._notify_status_change(old_status, self.system_status)
    
    async def _notify_status_change(self, old_status: SystemHealth, new_status: SystemHealth):
        """Notify about system status changes."""
        crypto_logger.logger.info(f"System health changed: {old_status.value} → {new_status.value}")
        
        # Create status change alert
        if new_status in [SystemHealth.CRITICAL, SystemHealth.WARNING]:
            priority = AlertPriority.HIGH if new_status == SystemHealth.CRITICAL else AlertPriority.WARNING
            alert = SystemAlert(
                id=f"status_change_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                priority=priority,
                category="system_health",
                title="System Status Change",
                message=f"System health changed to {new_status.value}",
                component="control_center"
            )
            await self.alert_manager.create_alert(alert)
        
        # Notify callbacks
        for callback in self._status_callbacks:
            try:
                await callback(old_status, new_status)
            except Exception as e:
                crypto_logger.logger.error(f"Status change callback failed: {e}")
    
    async def _handle_critical_alerts(self, alert: SystemAlert):
        """Handle critical alerts with automatic responses."""
        if alert.priority == AlertPriority.CRITICAL:
            # Implement automatic responses for critical alerts
            if "cpu" in alert.message.lower() or "memory" in alert.message.lower():
                # System resource alerts - enable conservative mode
                await self.override_manager.set_trading_override(
                    "conservative_mode", True, duration_minutes=30,
                    reason="Automatic response to system resource alert"
                )
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old data."""
        # This would run less frequently than the main loop
        # Clean up expired overrides, old alerts, etc.
        pass
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data."""
        latest_metrics = (
            self.performance_monitor.metrics_history[-1] 
            if self.performance_monitor.metrics_history 
            else None
        )
        
        dashboard = {
            "system_status": self.system_status.value,
            "timestamp": datetime.now(),
            "current_metrics": asdict(latest_metrics) if latest_metrics else {},
            "metrics_summary": self.performance_monitor.get_metrics_summary(),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "critical_alerts": len(self.alert_manager.get_active_alerts(AlertPriority.CRITICAL)),
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "active_overrides": len(self.override_manager.get_active_overrides()),
            "trading_control_status": trading_control_system.get_system_status(),
            "error_statistics": error_handler.get_error_statistics(),
            "validation_statistics": validation_engine.get_validation_statistics(),
            "black_box_statistics": black_box_processor.get_processing_statistics()
        }
        
        return dashboard
    
    def add_status_callback(self, callback: Callable):
        """Add system status change callback."""
        self._status_callbacks.append(callback)
    
    # Public API methods for external control
    async def emergency_stop(self, reason: str) -> str:
        """Trigger emergency stop."""
        return await self.override_manager.emergency_stop(reason)
    
    async def reset_circuit_breaker(self, breaker_name: str) -> bool:
        """Reset specific circuit breaker."""
        trading_control_system.circuit_breaker_manager.manually_reset_circuit_breaker(breaker_name)
        return True
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id)
    
    async def set_trading_restriction(self, restriction_type: str, duration_minutes: int = 60) -> str:
        """Set trading restriction."""
        return await self.override_manager.set_trading_override(
            "trading_restriction", restriction_type, duration_minutes,
            reason="Manual trading restriction"
        )


# Global control monitoring center instance
control_center = ControlMonitoringCenter()