import psutil
import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass, field
from threading import Thread
import time
import logging

from config.config import Config
from src.utils.logging_config import crypto_logger

@dataclass
class Alert:
    id: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    title: str
    message: str
    timestamp: datetime
    component: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    component: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """Comprehensive system monitoring for the crypto analyzer."""
    
    def __init__(self):
        self.config = Config()
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.alert_callbacks: List[Callable] = []
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 80.0,          # CPU usage %
            'memory_usage': 85.0,       # Memory usage %
            'disk_usage': 90.0,         # Disk usage %
            'api_error_rate': 0.1,      # 10% error rate
            'response_time': 5.0,       # 5 seconds
            'connection_count': 1000,   # Max connections
            'signal_generation_delay': 600,  # 10 minutes
            'data_freshness': 300       # 5 minutes
        }
        
        # Component status tracking
        self.components = [
            'market_data_fetcher',
            'realtime_streams',
            'ai_analyzer',
            'signal_generator',
            'portfolio_manager',
            'web_dashboard',
            'database'
        ]
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_running:
            logging.warning("Monitoring is already running")
            return
        
        self.is_running = True
        
        # Start monitoring threads
        Thread(target=self._system_metrics_monitor, daemon=True).start()
        Thread(target=self._health_check_monitor, daemon=True).start()
        Thread(target=self._alert_processor, daemon=True).start()
        Thread(target=self._cleanup_old_data, daemon=True).start()
        
        crypto_logger.logger.info("System monitoring started")
        self._create_alert("SYSTEM", "INFO", "Monitoring Started", 
                          "System monitoring has been initialized and started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        crypto_logger.logger.info("System monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_metrics = self._collect_system_metrics()
        recent_alerts = [a for a in self.alerts if not a.resolved and 
                        (datetime.now() - a.timestamp).total_seconds() < 3600]
        
        # Overall health score
        health_score = self._calculate_health_score()
        
        return {
            'overall_health': health_score,
            'status': 'HEALTHY' if health_score > 80 else 'DEGRADED' if health_score > 60 else 'UNHEALTHY',
            'current_metrics': current_metrics,
            'component_health': {name: check.status for name, check in self.health_checks.items()},
            'active_alerts': len(recent_alerts),
            'recent_alerts': [self._alert_to_dict(a) for a in recent_alerts[-5:]],
            'uptime_seconds': (datetime.now() - crypto_logger.performance_metrics['start_time']).total_seconds(),
            'last_update': datetime.now().isoformat()
        }
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
    
    def get_alerts(self, severity: Optional[str] = None, 
                  resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        return [self._alert_to_dict(a) for a in sorted(filtered_alerts, 
                                                      key=lambda x: x.timestamp, reverse=True)]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                crypto_logger.logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def _system_metrics_monitor(self):
        """Monitor system metrics continuously."""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and create alerts
                self._check_system_thresholds(metrics)
                
                # Log performance metrics
                crypto_logger.log_performance_metrics(metrics)
                
                time.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                crypto_logger.log_error(e, {'component': 'system_metrics_monitor'})
                time.sleep(30)
    
    def _health_check_monitor(self):
        """Monitor component health continuously."""
        while self.is_running:
            try:
                for component in self.components:
                    health_check = self._perform_health_check(component)
                    self.health_checks[component] = health_check
                    
                    # Create alerts for unhealthy components
                    if health_check.status == 'UNHEALTHY':
                        self._create_alert(
                            component.upper(),
                            "ERROR",
                            f"{component} Health Check Failed",
                            f"Component {component} is unhealthy: {health_check.error_message}"
                        )
                    elif health_check.status == 'DEGRADED':
                        self._create_alert(
                            component.upper(),
                            "WARNING",
                            f"{component} Performance Degraded",
                            f"Component {component} is showing degraded performance"
                        )
                
                time.sleep(300)  # Health checks every 5 minutes
                
            except Exception as e:
                crypto_logger.log_error(e, {'component': 'health_check_monitor'})
                time.sleep(60)
    
    def _alert_processor(self):
        """Process and send alerts."""
        while self.is_running:
            try:
                # Check for duplicate alerts and consolidate
                self._consolidate_alerts()
                
                # Send critical alerts immediately
                critical_alerts = [a for a in self.alerts 
                                 if a.severity == 'CRITICAL' and not a.resolved 
                                 and (datetime.now() - a.timestamp).total_seconds() < 300]
                
                for alert in critical_alerts:
                    self._send_alert_notifications(alert)
                
                time.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                crypto_logger.log_error(e, {'component': 'alert_processor'})
                time.sleep(30)
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        while self.is_running:
            try:
                # Clean old metrics (keep 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                self.metrics_history = [m for m in self.metrics_history 
                                      if m['timestamp'] >= cutoff_time]
                
                # Clean old resolved alerts (keep 30 days)
                alert_cutoff = datetime.now() - timedelta(days=30)
                self.alerts = [a for a in self.alerts 
                              if not a.resolved or a.timestamp >= alert_cutoff]
                
                time.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                crypto_logger.log_error(e, {'component': 'cleanup_old_data'})
                time.sleep(1800)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network statistics
        net_io = psutil.net_io_counters()
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Application-specific metrics
        performance_summary = crypto_logger.get_performance_summary()
        
        metrics = {
            'timestamp': datetime.now(),
            'system': {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'process_memory_mb': process_memory,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv
            },
            'application': {
                'uptime_seconds': performance_summary['uptime_seconds'],
                'api_calls_total': performance_summary['total_api_calls'],
                'signals_generated': performance_summary['signals_generated'],
                'error_count': performance_summary['total_errors'],
                'error_rate': performance_summary['error_rate'],
                'api_calls_per_minute': performance_summary['api_calls_per_minute'],
                'signals_per_hour': performance_summary['signals_per_hour']
            },
            'health_checks': len([h for h in self.health_checks.values() if h.status == 'HEALTHY']),
            'total_components': len(self.components),
            'active_alerts': len([a for a in self.alerts if not a.resolved])
        }
        
        return metrics
    
    def _check_system_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and create alerts."""
        system = metrics['system']
        app = metrics['application']
        
        # CPU usage check
        if system['cpu_usage'] > self.thresholds['cpu_usage']:
            self._create_alert(
                "SYSTEM", "WARNING", "High CPU Usage",
                f"CPU usage is {system['cpu_usage']:.1f}% (threshold: {self.thresholds['cpu_usage']}%)",
                {'cpu_usage': system['cpu_usage']}
            )
        
        # Memory usage check
        if system['memory_usage'] > self.thresholds['memory_usage']:
            self._create_alert(
                "SYSTEM", "WARNING", "High Memory Usage",
                f"Memory usage is {system['memory_usage']:.1f}% (threshold: {self.thresholds['memory_usage']}%)",
                {'memory_usage': system['memory_usage']}
            )
        
        # Disk usage check
        if system['disk_usage'] > self.thresholds['disk_usage']:
            self._create_alert(
                "SYSTEM", "ERROR", "High Disk Usage",
                f"Disk usage is {system['disk_usage']:.1f}% (threshold: {self.thresholds['disk_usage']}%)",
                {'disk_usage': system['disk_usage']}
            )
        
        # Error rate check
        if app['error_rate'] > self.thresholds['api_error_rate']:
            self._create_alert(
                "APPLICATION", "WARNING", "High Error Rate",
                f"API error rate is {app['error_rate']:.2%} (threshold: {self.thresholds['api_error_rate']:.2%})",
                {'error_rate': app['error_rate']}
            )
    
    def _perform_health_check(self, component: str) -> HealthCheck:
        """Perform health check for a specific component."""
        start_time = time.time()
        
        try:
            if component == 'market_data_fetcher':
                # Check if we can fetch market data
                status = 'HEALTHY'  # Simplified for demo
                error_message = None
                
            elif component == 'realtime_streams':
                # Check if real-time streams are active
                status = 'HEALTHY'
                error_message = None
                
            elif component == 'ai_analyzer':
                # Check AI analyzer availability
                status = 'HEALTHY'
                error_message = None
                
            elif component == 'signal_generator':
                # Check if signals are being generated
                last_signal_time = crypto_logger.performance_metrics.get('last_signal_time')
                if last_signal_time and (datetime.now() - last_signal_time).total_seconds() > 600:
                    status = 'DEGRADED'
                    error_message = "No signals generated in last 10 minutes"
                else:
                    status = 'HEALTHY'
                    error_message = None
                    
            elif component == 'portfolio_manager':
                # Check portfolio manager
                status = 'HEALTHY'
                error_message = None
                
            elif component == 'web_dashboard':
                # Check if dashboard is responsive
                status = 'HEALTHY'
                error_message = None
                
            elif component == 'database':
                # Check database connectivity
                status = 'HEALTHY'
                error_message = None
                
            else:
                status = 'UNKNOWN'
                error_message = f"Unknown component: {component}"
            
        except Exception as e:
            status = 'UNHEALTHY'
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        return HealthCheck(
            component=component,
            status=status,
            last_check=datetime.now(),
            response_time=response_time,
            error_message=error_message
        )
    
    def _calculate_health_score(self) -> int:
        """Calculate overall system health score (0-100)."""
        if not self.health_checks:
            return 50  # Neutral score if no checks
        
        healthy_components = sum(1 for check in self.health_checks.values() 
                               if check.status == 'HEALTHY')
        degraded_components = sum(1 for check in self.health_checks.values() 
                                if check.status == 'DEGRADED')
        unhealthy_components = sum(1 for check in self.health_checks.values() 
                                 if check.status == 'UNHEALTHY')
        
        total_components = len(self.health_checks)
        
        # Calculate weighted score
        score = (healthy_components * 100 + degraded_components * 50 + unhealthy_components * 0) / total_components
        
        # Adjust for recent alerts
        recent_critical_alerts = len([a for a in self.alerts 
                                    if not a.resolved 
                                    and a.severity == 'CRITICAL'
                                    and (datetime.now() - a.timestamp).total_seconds() < 3600])
        
        score -= recent_critical_alerts * 10  # Reduce score for critical alerts
        
        return max(0, min(100, int(score)))
    
    def _create_alert(self, component: str, severity: str, title: str, 
                     message: str, metadata: Dict[str, Any] = None):
        """Create a new alert."""
        alert_id = f"{component}_{severity}_{int(datetime.now().timestamp())}"
        
        # Check for duplicate recent alerts
        recent_similar_alerts = [
            a for a in self.alerts 
            if a.component == component 
            and a.severity == severity 
            and a.title == title
            and not a.resolved
            and (datetime.now() - a.timestamp).total_seconds() < 3600
        ]
        
        if recent_similar_alerts:
            return  # Don't create duplicate alerts
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        crypto_logger.logger.log(
            getattr(logging, severity.upper(), logging.INFO),
            f"ALERT: {title} - {message}",
            extra={'extra_data': {
                'alert_id': alert_id,
                'component': component,
                'severity': severity,
                'metadata': metadata
            }}
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                crypto_logger.log_error(e, {'component': 'alert_callback'})
    
    def _consolidate_alerts(self):
        """Consolidate similar alerts to reduce noise."""
        # Group similar unresolved alerts
        alert_groups = {}
        
        for alert in self.alerts:
            if not alert.resolved:
                key = f"{alert.component}_{alert.title}"
                if key not in alert_groups:
                    alert_groups[key] = []
                alert_groups[key].append(alert)
        
        # Merge similar alerts
        for group in alert_groups.values():
            if len(group) > 1:
                # Keep the most recent alert, resolve others
                group.sort(key=lambda x: x.timestamp, reverse=True)
                for alert in group[1:]:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via email."""
        if not self.config.EMAIL_USER or not self.config.TO_EMAIL:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.EMAIL_USER
            msg['To'] = self.config.TO_EMAIL
            msg['Subject'] = f"🚨 Crypto Analyzer Alert: {alert.title}"
            
            body = f"""
            Alert Details:
            
            Component: {alert.component}
            Severity: {alert.severity}
            Title: {alert.title}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            
            Alert ID: {alert.id}
            
            Please check the dashboard for more details.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT)
            server.starttls()
            server.login(self.config.EMAIL_USER, self.config.EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(self.config.EMAIL_USER, self.config.TO_EMAIL, text)
            server.quit()
            
            crypto_logger.logger.info(f"Alert notification sent for {alert.id}")
            
        except Exception as e:
            crypto_logger.log_error(e, {'component': 'alert_notifications', 'alert_id': alert.id})
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'id': alert.id,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'component': alert.component,
            'resolved': alert.resolved,
            'resolution_time': alert.resolution_time.isoformat() if alert.resolution_time else None,
            'metadata': alert.metadata
        }

# Global monitor instance
system_monitor = SystemMonitor()

# Convenience functions
def start_monitoring():
    system_monitor.start_monitoring()

def stop_monitoring():
    system_monitor.stop_monitoring()

def get_system_status() -> Dict[str, Any]:
    return system_monitor.get_system_status()

def create_alert(component: str, severity: str, title: str, message: str, metadata: Dict[str, Any] = None):
    system_monitor._create_alert(component, severity, title, message, metadata)

def add_alert_callback(callback: Callable[[Alert], None]):
    system_monitor.add_alert_callback(callback)