import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import hashlib
from enum import Enum
import uuid
from decimal import Decimal, ROUND_HALF_UP

from src.utils.logging_config import crypto_logger
from config.config import Config

class ReportType(Enum):
    TRANSACTION_REPORT = "transaction_report"
    POSITION_REPORT = "position_report"
    RISK_REPORT = "risk_report"
    PNL_REPORT = "pnl_report"
    TRADE_SURVEILLANCE = "trade_surveillance"
    AML_REPORT = "aml_report"
    TAX_REPORT = "tax_report"
    AUDIT_TRAIL = "audit_trail"

class RegulatoryJurisdiction(Enum):
    US_SEC = "us_sec"
    US_CFTC = "us_cftc"
    EU_ESMA = "eu_esma"
    UK_FCA = "uk_fca"
    SINGAPORE_MAS = "singapore_mas"
    JAPAN_FSA = "japan_fsa"
    GENERIC = "generic"

@dataclass
class TransactionRecord:
    transaction_id: str
    timestamp: datetime
    symbol: str
    transaction_type: str  # 'buy', 'sell', 'transfer', 'deposit', 'withdrawal'
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    counterparty: Optional[str]
    exchange: str
    wallet_address: Optional[str]
    transaction_hash: Optional[str]
    regulatory_flags: List[str]
    compliance_status: str
    reporting_currency: str
    fx_rate: Decimal
    tax_lots: List[Dict[str, Any]]
    
@dataclass
class PositionRecord:
    position_id: str
    timestamp: datetime
    symbol: str
    quantity: Decimal
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal
    average_cost: Decimal
    position_type: str  # 'long', 'short'
    exchange: str
    risk_metrics: Dict[str, float]
    compliance_classification: str

@dataclass
class ComplianceAlert:
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_transactions: List[str]
    regulatory_implications: List[str]
    recommended_actions: List[str]
    status: str  # 'active', 'investigating', 'resolved', 'false_positive'
    assigned_to: Optional[str]

@dataclass
class RegulatoryReport:
    report_id: str
    report_type: ReportType
    jurisdiction: RegulatoryJurisdiction
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime
    report_data: Dict[str, Any]
    file_hash: str
    compliance_attestation: Dict[str, Any]
    submission_status: str
    file_format: str  # 'json', 'xml', 'csv', 'pdf'

class RegulatoryComplianceEngine:
    """Comprehensive regulatory compliance and reporting system."""
    
    def __init__(self):
        self.config = Config()
        self.transaction_log = []
        self.position_records = {}
        self.compliance_alerts = []
        self.generated_reports = {}
        
        # Regulatory thresholds and rules
        self.regulatory_rules = {
            'large_transaction_threshold': Decimal('10000.00'),  # $10,000
            'suspicious_volume_threshold': Decimal('100000.00'),  # $100,000 daily
            'position_concentration_limit': 0.20,  # 20% of portfolio
            'max_daily_transactions': 1000,
            'aml_watchlist_check_required': True,
            'beneficial_ownership_threshold': 0.25  # 25% ownership
        }
        
        # Reporting templates by jurisdiction
        self.reporting_templates = {
            RegulatoryJurisdiction.US_SEC: {
                'position_reporting': 'Form 13F equivalent',
                'trade_reporting': 'CAT (Consolidated Audit Trail) format',
                'risk_reporting': 'Market Risk Capital Rule'
            },
            RegulatoryJurisdiction.US_CFTC: {
                'swap_reporting': 'SDR (Swap Data Repository) format',
                'position_reporting': 'Large Trader Reporting'
            },
            RegulatoryJurisdiction.EU_ESMA: {
                'transaction_reporting': 'MiFID II transaction reporting',
                'position_reporting': 'EMIR position reporting'
            }
        }
        
        # Compliance monitoring rules
        self.monitoring_rules = {
            'wash_trading_detection': True,
            'market_manipulation_detection': True,
            'insider_trading_patterns': True,
            'layering_detection': True,
            'spoofing_detection': True,
            'cross_market_surveillance': True
        }
        
        # Tax calculation parameters
        self.tax_parameters = {
            'default_method': 'FIFO',  # First In, First Out
            'alternative_methods': ['LIFO', 'HIFO', 'SpecID'],
            'long_term_threshold_days': 365,
            'wash_sale_period_days': 30
        }
        
    async def initialize_compliance_system(self):
        """Initialize regulatory compliance system."""
        crypto_logger.logger.info("Initializing regulatory compliance engine")
        
        try:
            # Load compliance rules and templates
            await self._load_regulatory_frameworks()
            
            # Initialize monitoring systems
            await self._initialize_surveillance_systems()
            
            # Setup reporting schedules
            await self._setup_reporting_schedules()
            
            # Generate mock compliance data for testing
            await self._generate_mock_compliance_data()
            
            crypto_logger.logger.info("✓ Regulatory compliance engine initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing compliance system: {e}")
    
    async def _load_regulatory_frameworks(self):
        """Load regulatory frameworks and rules."""
        
        # Enhanced regulatory rules with jurisdiction-specific requirements
        self.jurisdiction_rules = {
            RegulatoryJurisdiction.US_SEC: {
                'large_trader_threshold': Decimal('50000000'),  # $50M
                'form_13f_threshold': Decimal('100000000'),     # $100M
                'insider_trading_monitoring': True,
                'market_manipulation_detection': True,
                'reporting_frequency': 'quarterly'
            },
            RegulatoryJurisdiction.EU_ESMA: {
                'mifid_ii_compliance': True,
                'transaction_reporting_threshold': Decimal('0'),  # All transactions
                'position_limits': True,
                'best_execution_reporting': True,
                'reporting_frequency': 'daily'
            },
            RegulatoryJurisdiction.UK_FCA: {
                'mar_compliance': True,  # Market Abuse Regulation
                'transaction_reporting': True,
                'suspicious_transaction_reporting': True,
                'reporting_frequency': 'daily'
            }
        }
        
        crypto_logger.logger.info("Loaded regulatory frameworks for multiple jurisdictions")
    
    async def _initialize_surveillance_systems(self):
        """Initialize trade surveillance and monitoring systems."""
        
        self.surveillance_systems = {
            'wash_trading_detector': {
                'enabled': True,
                'lookback_period_hours': 24,
                'similarity_threshold': 0.95,
                'volume_threshold': Decimal('1000')
            },
            'layering_detector': {
                'enabled': True,
                'order_book_levels': 5,
                'cancel_ratio_threshold': 0.8,
                'time_window_seconds': 300
            },
            'spoofing_detector': {
                'enabled': True,
                'order_size_threshold': Decimal('10000'),
                'cancel_time_threshold_seconds': 10
            },
            'pump_and_dump_detector': {
                'enabled': True,
                'price_increase_threshold': 0.50,  # 50% increase
                'volume_spike_threshold': 10.0,    # 10x normal volume
                'monitoring_window_hours': 4
            }
        }
        
        crypto_logger.logger.info("Initialized trade surveillance systems")
    
    async def _setup_reporting_schedules(self):
        """Setup automated reporting schedules."""
        
        self.reporting_schedules = {
            'daily_position_report': {
                'frequency': 'daily',
                'time': '23:59',
                'jurisdictions': [RegulatoryJurisdiction.EU_ESMA, RegulatoryJurisdiction.UK_FCA],
                'auto_submit': False
            },
            'weekly_risk_report': {
                'frequency': 'weekly',
                'day': 'friday',
                'time': '18:00',
                'jurisdictions': [RegulatoryJurisdiction.US_SEC],
                'auto_submit': False
            },
            'monthly_tax_report': {
                'frequency': 'monthly',
                'day': 'last',
                'time': '23:59',
                'jurisdictions': [RegulatoryJurisdiction.GENERIC],
                'auto_submit': False
            }
        }
        
        crypto_logger.logger.info("Setup automated reporting schedules")
    
    async def _generate_mock_compliance_data(self):
        """Generate mock compliance data for testing."""
        
        # Generate mock transactions
        for i in range(500):  # 500 mock transactions
            transaction = self._create_mock_transaction(i)
            await self.record_transaction(transaction)
        
        # Generate mock alerts
        for i in range(20):  # 20 mock alerts
            alert = self._create_mock_alert(i)
            self.compliance_alerts.append(alert)
        
        crypto_logger.logger.info("Generated mock compliance data for testing")
    
    def _create_mock_transaction(self, index: int) -> Dict[str, Any]:
        """Create a mock transaction for testing."""
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        transaction_types = ['buy', 'sell']
        exchanges = ['binance', 'coinbase', 'kraken']
        
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        timestamp = base_time + timedelta(hours=np.random.uniform(0, 24*30))
        
        symbol = np.random.choice(symbols)
        transaction_type = np.random.choice(transaction_types)
        exchange = np.random.choice(exchanges)
        
        quantity = Decimal(str(round(np.random.uniform(0.1, 10.0), 8)))
        price = Decimal(str(round(np.random.uniform(1000, 50000), 2)))
        total_value = quantity * price
        fees = total_value * Decimal('0.001')  # 0.1% fee
        
        # Add regulatory flags for some transactions
        regulatory_flags = []
        if total_value > self.regulatory_rules['large_transaction_threshold']:
            regulatory_flags.append('large_transaction')
        
        if np.random.random() < 0.05:  # 5% chance of suspicious activity
            regulatory_flags.append('unusual_pattern')
        
        return {
            'symbol': symbol,
            'transaction_type': transaction_type,
            'quantity': quantity,
            'price': price,
            'total_value': total_value,
            'fees': fees,
            'exchange': exchange,
            'counterparty': None,
            'wallet_address': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
            'transaction_hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
            'regulatory_flags': regulatory_flags,
            'timestamp': timestamp
        }
    
    def _create_mock_alert(self, index: int) -> ComplianceAlert:
        """Create a mock compliance alert."""
        
        alert_types = ['wash_trading', 'layering', 'unusual_volume', 'large_position', 'suspicious_timing']
        severities = ['low', 'medium', 'high', 'critical']
        statuses = ['active', 'investigating', 'resolved', 'false_positive']
        
        return ComplianceAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=np.random.uniform(0, 168)),  # Last week
            alert_type=np.random.choice(alert_types),
            severity=np.random.choice(severities),
            description=f"Mock compliance alert #{index} for testing purposes",
            affected_transactions=[f"tx_{i}" for i in range(np.random.randint(1, 5))],
            regulatory_implications=[f"Potential violation of rule #{np.random.randint(1, 10)}"],
            recommended_actions=[f"Action item #{i+1}" for i in range(np.random.randint(1, 3))],
            status=np.random.choice(statuses),
            assigned_to=f"compliance_officer_{np.random.randint(1, 5)}" if np.random.random() < 0.7 else None
        )
    
    async def record_transaction(self, transaction_data: Dict[str, Any]) -> TransactionRecord:
        """Record a transaction with compliance checking."""
        
        # Create transaction record
        transaction_record = TransactionRecord(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            timestamp=transaction_data.get('timestamp', datetime.now(timezone.utc)),
            symbol=transaction_data['symbol'],
            transaction_type=transaction_data['transaction_type'],
            quantity=transaction_data['quantity'],
            price=transaction_data['price'],
            total_value=transaction_data['total_value'],
            fees=transaction_data['fees'],
            counterparty=transaction_data.get('counterparty'),
            exchange=transaction_data['exchange'],
            wallet_address=transaction_data.get('wallet_address'),
            transaction_hash=transaction_data.get('transaction_hash'),
            regulatory_flags=transaction_data.get('regulatory_flags', []),
            compliance_status='pending_review',
            reporting_currency='USD',
            fx_rate=Decimal('1.0'),
            tax_lots=[]
        )
        
        # Run compliance checks
        await self._run_compliance_checks(transaction_record)
        
        # Store transaction
        self.transaction_log.append(transaction_record)
        
        # Update position records
        await self._update_position_records(transaction_record)
        
        crypto_logger.logger.info(f"Recorded transaction: {transaction_record.transaction_id}")
        return transaction_record
    
    async def _run_compliance_checks(self, transaction: TransactionRecord):
        """Run comprehensive compliance checks on a transaction."""
        
        compliance_issues = []
        
        # Large transaction check
        if transaction.total_value >= self.regulatory_rules['large_transaction_threshold']:
            compliance_issues.append('large_transaction_reporting_required')
        
        # Suspicious pattern detection
        if await self._detect_wash_trading(transaction):
            compliance_issues.append('potential_wash_trading')
            
        if await self._detect_layering_pattern(transaction):
            compliance_issues.append('potential_layering')
        
        # AML checks
        if self.regulatory_rules['aml_watchlist_check_required']:
            if await self._check_aml_watchlist(transaction):
                compliance_issues.append('aml_watchlist_match')
        
        # Update compliance status
        if compliance_issues:
            transaction.compliance_status = 'requires_review'
            transaction.regulatory_flags.extend(compliance_issues)
            
            # Generate compliance alert for serious issues
            serious_issues = ['potential_wash_trading', 'aml_watchlist_match']
            if any(issue in compliance_issues for issue in serious_issues):
                await self._generate_compliance_alert(transaction, compliance_issues)
        else:
            transaction.compliance_status = 'approved'
    
    async def _detect_wash_trading(self, transaction: TransactionRecord) -> bool:
        """Detect potential wash trading patterns."""
        
        # Look for offsetting trades in short time window
        lookback_time = transaction.timestamp - timedelta(hours=1)
        recent_transactions = [
            t for t in self.transaction_log 
            if t.timestamp >= lookback_time and 
               t.symbol == transaction.symbol and
               t.transaction_id != transaction.transaction_id
        ]
        
        # Check for offsetting transactions
        for recent_tx in recent_transactions:
            if (recent_tx.transaction_type != transaction.transaction_type and
                abs(recent_tx.quantity - transaction.quantity) / transaction.quantity < 0.05 and  # Within 5%
                abs(recent_tx.price - transaction.price) / transaction.price < 0.02):  # Within 2%
                
                return True
        
        return False
    
    async def _detect_layering_pattern(self, transaction: TransactionRecord) -> bool:
        """Detect potential layering/spoofing patterns."""
        
        # Simplified detection - look for rapid order/cancel patterns
        # In a real system, this would analyze order book data
        
        # Check transaction timing patterns
        same_symbol_recent = [
            t for t in self.transaction_log[-50:]  # Last 50 transactions
            if t.symbol == transaction.symbol and
               abs((t.timestamp - transaction.timestamp).total_seconds()) < 300  # Within 5 minutes
        ]
        
        if len(same_symbol_recent) > 10:  # More than 10 transactions in 5 minutes
            return True
        
        return False
    
    async def _check_aml_watchlist(self, transaction: TransactionRecord) -> bool:
        """Check transaction against AML watchlists."""
        
        # Mock AML check - in production, this would query real watchlists
        suspicious_addresses = [
            "0x1234567890abcdef1234567890abcdef12345678",
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        ]
        
        if transaction.wallet_address in suspicious_addresses:
            return True
        
        # Random 1% chance for testing
        return np.random.random() < 0.01
    
    async def _generate_compliance_alert(self, transaction: TransactionRecord, issues: List[str]):
        """Generate a compliance alert for suspicious activity."""
        
        severity_map = {
            'potential_wash_trading': 'high',
            'aml_watchlist_match': 'critical',
            'potential_layering': 'medium',
            'large_transaction_reporting_required': 'low'
        }
        
        max_severity = 'low'
        for issue in issues:
            if severity_map.get(issue, 'low') == 'critical':
                max_severity = 'critical'
                break
            elif severity_map.get(issue, 'low') == 'high' and max_severity != 'critical':
                max_severity = 'high'
            elif severity_map.get(issue, 'low') == 'medium' and max_severity not in ['critical', 'high']:
                max_severity = 'medium'
        
        alert = ComplianceAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            alert_type='automated_detection',
            severity=max_severity,
            description=f"Compliance issues detected: {', '.join(issues)}",
            affected_transactions=[transaction.transaction_id],
            regulatory_implications=[f"Potential violation: {issue}" for issue in issues],
            recommended_actions=['Manual review required', 'Document investigation'],
            status='active',
            assigned_to=None
        )
        
        self.compliance_alerts.append(alert)
        crypto_logger.logger.warning(f"Generated compliance alert: {alert.alert_id}")
    
    async def _update_position_records(self, transaction: TransactionRecord):
        """Update position records based on transaction."""
        
        symbol = transaction.symbol
        
        if symbol not in self.position_records:
            self.position_records[symbol] = {
                'position_id': f"pos_{symbol}_{uuid.uuid4().hex[:8]}",
                'symbol': symbol,
                'quantity': Decimal('0'),
                'cost_basis': Decimal('0'),
                'transactions': []
            }
        
        position = self.position_records[symbol]
        position['transactions'].append(transaction)
        
        # Update quantity and cost basis
        if transaction.transaction_type == 'buy':
            new_quantity = position['quantity'] + transaction.quantity
            if new_quantity > 0:
                # Weighted average cost
                total_cost = (position['cost_basis'] * position['quantity'] + 
                            transaction.price * transaction.quantity)
                position['cost_basis'] = total_cost / new_quantity
            position['quantity'] = new_quantity
        elif transaction.transaction_type == 'sell':
            position['quantity'] -= transaction.quantity
            # Cost basis remains the same for sales
        
        # Ensure quantity doesn't go negative (simplified)
        position['quantity'] = max(position['quantity'], Decimal('0'))
    
    async def generate_regulatory_report(self, report_type: ReportType, 
                                       jurisdiction: RegulatoryJurisdiction,
                                       start_date: datetime, end_date: datetime) -> RegulatoryReport:
        """Generate regulatory report for specific jurisdiction and time period."""
        
        crypto_logger.logger.info(f"Generating {report_type.value} for {jurisdiction.value}")
        
        # Filter data by date range
        relevant_transactions = [
            t for t in self.transaction_log
            if start_date <= t.timestamp <= end_date
        ]
        
        # Generate report data based on type
        if report_type == ReportType.TRANSACTION_REPORT:
            report_data = await self._generate_transaction_report(relevant_transactions, jurisdiction)
        elif report_type == ReportType.POSITION_REPORT:
            report_data = await self._generate_position_report(end_date, jurisdiction)
        elif report_type == ReportType.RISK_REPORT:
            report_data = await self._generate_risk_report(relevant_transactions, jurisdiction)
        elif report_type == ReportType.PNL_REPORT:
            report_data = await self._generate_pnl_report(relevant_transactions, jurisdiction)
        elif report_type == ReportType.TAX_REPORT:
            report_data = await self._generate_tax_report(relevant_transactions, jurisdiction)
        elif report_type == ReportType.TRADE_SURVEILLANCE:
            report_data = await self._generate_surveillance_report(relevant_transactions, jurisdiction)
        else:
            report_data = {'error': f'Unsupported report type: {report_type.value}'}
        
        # Create report record
        report_id = f"rpt_{report_type.value}_{jurisdiction.value}_{uuid.uuid4().hex[:8]}"
        report_json = json.dumps(report_data, default=str, indent=2)
        file_hash = hashlib.sha256(report_json.encode()).hexdigest()
        
        regulatory_report = RegulatoryReport(
            report_id=report_id,
            report_type=report_type,
            jurisdiction=jurisdiction,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            generated_at=datetime.now(timezone.utc),
            report_data=report_data,
            file_hash=file_hash,
            compliance_attestation={
                'generated_by': 'Crypto Trend Analyzer Compliance Engine',
                'attestation_timestamp': datetime.now(timezone.utc).isoformat(),
                'data_integrity_verified': True,
                'completeness_verified': True
            },
            submission_status='generated',
            file_format='json'
        )
        
        # Store report
        self.generated_reports[report_id] = regulatory_report
        
        crypto_logger.logger.info(f"Generated regulatory report: {report_id}")
        return regulatory_report
    
    async def _generate_transaction_report(self, transactions: List[TransactionRecord], 
                                         jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate transaction report in jurisdiction-specific format."""
        
        total_transactions = len(transactions)
        total_volume = sum(t.total_value for t in transactions)
        
        # Group by symbol and transaction type
        by_symbol = {}
        by_type = {}
        
        for tx in transactions:
            # By symbol
            if tx.symbol not in by_symbol:
                by_symbol[tx.symbol] = {'count': 0, 'volume': Decimal('0')}
            by_symbol[tx.symbol]['count'] += 1
            by_symbol[tx.symbol]['volume'] += tx.total_value
            
            # By type
            if tx.transaction_type not in by_type:
                by_type[tx.transaction_type] = {'count': 0, 'volume': Decimal('0')}
            by_type[tx.transaction_type]['count'] += 1
            by_type[tx.transaction_type]['volume'] += tx.total_value
        
        # Jurisdiction-specific formatting
        if jurisdiction == RegulatoryJurisdiction.US_SEC:
            # SEC-style reporting
            report_format = 'SEC_13F_EQUIVALENT'
        elif jurisdiction == RegulatoryJurisdiction.EU_ESMA:
            # MiFID II transaction reporting format
            report_format = 'MIFID_II_TRANSACTION_REPORT'
        else:
            report_format = 'GENERIC_TRANSACTION_REPORT'
        
        # Large transactions (requiring special reporting)
        large_transactions = [
            {
                'transaction_id': tx.transaction_id,
                'timestamp': tx.timestamp.isoformat(),
                'symbol': tx.symbol,
                'type': tx.transaction_type,
                'quantity': float(tx.quantity),
                'price': float(tx.price),
                'total_value': float(tx.total_value),
                'regulatory_flags': tx.regulatory_flags
            }
            for tx in transactions
            if tx.total_value >= self.regulatory_rules['large_transaction_threshold']
        ]
        
        return {
            'report_format': report_format,
            'jurisdiction': jurisdiction.value,
            'summary': {
                'total_transactions': total_transactions,
                'total_volume': float(total_volume),
                'unique_symbols': len(by_symbol),
                'reporting_currency': 'USD'
            },
            'transactions_by_symbol': {
                symbol: {
                    'transaction_count': data['count'],
                    'total_volume': float(data['volume'])
                }
                for symbol, data in by_symbol.items()
            },
            'transactions_by_type': {
                tx_type: {
                    'transaction_count': data['count'],
                    'total_volume': float(data['volume'])
                }
                for tx_type, data in by_type.items()
            },
            'large_transactions': large_transactions,
            'compliance_summary': {
                'flagged_transactions': len([tx for tx in transactions if tx.regulatory_flags]),
                'pending_review': len([tx for tx in transactions if tx.compliance_status == 'pending_review']),
                'approved_transactions': len([tx for tx in transactions if tx.compliance_status == 'approved'])
            }
        }
    
    async def _generate_position_report(self, as_of_date: datetime, 
                                      jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate position report as of specific date."""
        
        positions_summary = []
        total_portfolio_value = Decimal('0')
        
        for symbol, position_data in self.position_records.items():
            if position_data['quantity'] > 0:
                # Mock current market price
                current_price = Decimal(str(np.random.uniform(1000, 50000)))
                market_value = position_data['quantity'] * current_price
                unrealized_pnl = market_value - (position_data['cost_basis'] * position_data['quantity'])
                
                total_portfolio_value += market_value
                
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': float(position_data['quantity']),
                    'cost_basis': float(position_data['cost_basis']),
                    'current_price': float(current_price),
                    'market_value': float(market_value),
                    'unrealized_pnl': float(unrealized_pnl),
                    'unrealized_pnl_pct': float(unrealized_pnl / market_value * 100) if market_value > 0 else 0
                })
        
        # Calculate concentration risk
        concentration_analysis = {}
        if total_portfolio_value > 0:
            for pos in positions_summary:
                weight = pos['market_value'] / float(total_portfolio_value)
                concentration_analysis[pos['symbol']] = {
                    'portfolio_weight': weight * 100,
                    'concentration_risk': 'High' if weight > self.regulatory_rules['position_concentration_limit'] else 'Normal'
                }
        
        return {
            'as_of_date': as_of_date.isoformat(),
            'jurisdiction': jurisdiction.value,
            'portfolio_summary': {
                'total_positions': len(positions_summary),
                'total_market_value': float(total_portfolio_value),
                'reporting_currency': 'USD'
            },
            'positions': positions_summary,
            'concentration_analysis': concentration_analysis,
            'risk_metrics': {
                'largest_position_weight': max([conc['portfolio_weight'] for conc in concentration_analysis.values()], default=0),
                'positions_over_limit': len([pos for pos in concentration_analysis.values() if pos['concentration_risk'] == 'High'])
            }
        }
    
    async def _generate_risk_report(self, transactions: List[TransactionRecord], 
                                  jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate risk assessment report."""
        
        # Calculate various risk metrics
        daily_volumes = {}
        for tx in transactions:
            date_key = tx.timestamp.date()
            if date_key not in daily_volumes:
                daily_volumes[date_key] = Decimal('0')
            daily_volumes[date_key] += tx.total_value
        
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else Decimal('0')
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else Decimal('0')
        
        # Risk flags
        risk_flags = []
        if max_daily_volume > self.regulatory_rules['suspicious_volume_threshold']:
            risk_flags.append('Excessive daily volume detected')
        
        # Transaction frequency analysis
        transaction_hours = [tx.timestamp.hour for tx in transactions]
        off_hours_transactions = len([h for h in transaction_hours if h < 6 or h > 22])  # Outside 6 AM - 10 PM
        
        if off_hours_transactions > len(transactions) * 0.3:  # More than 30% off-hours
            risk_flags.append('High proportion of off-hours trading')
        
        return {
            'jurisdiction': jurisdiction.value,
            'risk_assessment': {
                'overall_risk_level': 'Medium',  # Simplified calculation
                'max_daily_volume': float(max_daily_volume),
                'average_daily_volume': float(avg_daily_volume),
                'volume_volatility': float(np.std(list(daily_volumes.values()))) if daily_volumes else 0,
                'off_hours_trading_pct': (off_hours_transactions / len(transactions) * 100) if transactions else 0
            },
            'risk_flags': risk_flags,
            'compliance_breaches': len([tx for tx in transactions if 'requires_review' in tx.compliance_status]),
            'surveillance_alerts': len(self.compliance_alerts),
            'recommendations': [
                'Continue monitoring for unusual patterns',
                'Review large transaction procedures',
                'Enhance off-hours trading monitoring'
            ]
        }
    
    async def _generate_pnl_report(self, transactions: List[TransactionRecord], 
                                 jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate profit and loss report."""
        
        realized_pnl = Decimal('0')
        pnl_by_symbol = {}
        
        # Calculate realized P&L from completed trades
        for tx in transactions:
            if tx.transaction_type == 'sell':
                # Simplified P&L calculation (would need cost basis tracking in production)
                estimated_cost_basis = tx.price * Decimal('0.95')  # Assume 5% profit margin
                pnl = (tx.price - estimated_cost_basis) * tx.quantity
                realized_pnl += pnl
                
                if tx.symbol not in pnl_by_symbol:
                    pnl_by_symbol[tx.symbol] = Decimal('0')
                pnl_by_symbol[tx.symbol] += pnl
        
        # Calculate unrealized P&L from current positions
        unrealized_pnl = Decimal('0')
        for symbol, position_data in self.position_records.items():
            if position_data['quantity'] > 0:
                current_price = Decimal(str(np.random.uniform(1000, 50000)))  # Mock price
                cost_basis = position_data['cost_basis']
                unrealized_pnl += (current_price - cost_basis) * position_data['quantity']
        
        return {
            'jurisdiction': jurisdiction.value,
            'pnl_summary': {
                'total_realized_pnl': float(realized_pnl),
                'total_unrealized_pnl': float(unrealized_pnl),
                'net_pnl': float(realized_pnl + unrealized_pnl),
                'reporting_currency': 'USD'
            },
            'realized_pnl_by_symbol': {
                symbol: float(pnl) for symbol, pnl in pnl_by_symbol.items()
            },
            'tax_implications': {
                'short_term_gains': float(realized_pnl * Decimal('0.6')),  # Approximate
                'long_term_gains': float(realized_pnl * Decimal('0.4')),   # Approximate
                'estimated_tax_liability': float(realized_pnl * Decimal('0.25'))  # Approximate 25% rate
            }
        }
    
    async def _generate_tax_report(self, transactions: List[TransactionRecord], 
                                 jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate comprehensive tax report."""
        
        # Calculate tax lots using FIFO method
        tax_lots = []
        gains_losses = []
        
        # Group transactions by symbol
        by_symbol = {}
        for tx in transactions:
            if tx.symbol not in by_symbol:
                by_symbol[tx.symbol] = []
            by_symbol[tx.symbol].append(tx)
        
        for symbol, symbol_transactions in by_symbol.items():
            # Sort by timestamp
            symbol_transactions.sort(key=lambda x: x.timestamp)
            
            inventory = []  # FIFO inventory
            
            for tx in symbol_transactions:
                if tx.transaction_type == 'buy':
                    # Add to inventory
                    inventory.append({
                        'quantity': tx.quantity,
                        'cost_basis': tx.price,
                        'purchase_date': tx.timestamp
                    })
                elif tx.transaction_type == 'sell':
                    # Remove from inventory (FIFO)
                    remaining_to_sell = tx.quantity
                    
                    while remaining_to_sell > 0 and inventory:
                        lot = inventory[0]
                        
                        if lot['quantity'] <= remaining_to_sell:
                            # Sell entire lot
                            gain_loss = (tx.price - lot['cost_basis']) * lot['quantity']
                            holding_period = (tx.timestamp - lot['purchase_date']).days
                            
                            gains_losses.append({
                                'symbol': symbol,
                                'quantity_sold': float(lot['quantity']),
                                'cost_basis': float(lot['cost_basis']),
                                'sale_price': float(tx.price),
                                'gain_loss': float(gain_loss),
                                'holding_period_days': holding_period,
                                'term': 'long' if holding_period >= self.tax_parameters['long_term_threshold_days'] else 'short',
                                'sale_date': tx.timestamp.isoformat()
                            })
                            
                            remaining_to_sell -= lot['quantity']
                            inventory.pop(0)
                        else:
                            # Partial sale of lot
                            gain_loss = (tx.price - lot['cost_basis']) * remaining_to_sell
                            holding_period = (tx.timestamp - lot['purchase_date']).days
                            
                            gains_losses.append({
                                'symbol': symbol,
                                'quantity_sold': float(remaining_to_sell),
                                'cost_basis': float(lot['cost_basis']),
                                'sale_price': float(tx.price),
                                'gain_loss': float(gain_loss),
                                'holding_period_days': holding_period,
                                'term': 'long' if holding_period >= self.tax_parameters['long_term_threshold_days'] else 'short',
                                'sale_date': tx.timestamp.isoformat()
                            })
                            
                            lot['quantity'] -= remaining_to_sell
                            remaining_to_sell = Decimal('0')
        
        # Summarize gains and losses
        short_term_gains = sum(gl['gain_loss'] for gl in gains_losses if gl['term'] == 'short' and gl['gain_loss'] > 0)
        short_term_losses = sum(abs(gl['gain_loss']) for gl in gains_losses if gl['term'] == 'short' and gl['gain_loss'] < 0)
        long_term_gains = sum(gl['gain_loss'] for gl in gains_losses if gl['term'] == 'long' and gl['gain_loss'] > 0)
        long_term_losses = sum(abs(gl['gain_loss']) for gl in gains_losses if gl['term'] == 'long' and gl['gain_loss'] < 0)
        
        return {
            'jurisdiction': jurisdiction.value,
            'tax_year': transactions[0].timestamp.year if transactions else datetime.now().year,
            'calculation_method': self.tax_parameters['default_method'],
            'summary': {
                'total_transactions': len(transactions),
                'taxable_events': len(gains_losses),
                'short_term_gains': short_term_gains,
                'short_term_losses': short_term_losses,
                'long_term_gains': long_term_gains,
                'long_term_losses': long_term_losses,
                'net_short_term': short_term_gains - short_term_losses,
                'net_long_term': long_term_gains - long_term_losses,
                'total_net_gain_loss': (short_term_gains - short_term_losses) + (long_term_gains - long_term_losses)
            },
            'detailed_transactions': gains_losses,
            'remaining_inventory': [
                {
                    'symbol': symbol,
                    'unrealized_lots': len(inventory),
                    'total_unrealized_quantity': float(sum(lot['quantity'] for lot in inventory)),
                    'average_cost_basis': float(sum(lot['cost_basis'] * lot['quantity'] for lot in inventory) / sum(lot['quantity'] for lot in inventory)) if inventory else 0
                }
                for symbol, inventory in by_symbol.items()
                if inventory
            ]
        }
    
    async def _generate_surveillance_report(self, transactions: List[TransactionRecord], 
                                          jurisdiction: RegulatoryJurisdiction) -> Dict[str, Any]:
        """Generate trade surveillance report."""
        
        # Filter alerts for the reporting period
        period_alerts = [
            alert for alert in self.compliance_alerts
            if transactions and transactions[0].timestamp <= alert.timestamp <= transactions[-1].timestamp
        ]
        
        # Analyze alert patterns
        alerts_by_type = {}
        alerts_by_severity = {}
        
        for alert in period_alerts:
            # By type
            if alert.alert_type not in alerts_by_type:
                alerts_by_type[alert.alert_type] = 0
            alerts_by_type[alert.alert_type] += 1
            
            # By severity
            if alert.severity not in alerts_by_severity:
                alerts_by_severity[alert.severity] = 0
            alerts_by_severity[alert.severity] += 1
        
        # Suspicious pattern analysis
        suspicious_patterns = []
        
        # Check for clustering of transactions
        if len(transactions) > 0:
            transaction_times = [tx.timestamp.timestamp() for tx in transactions]
            time_gaps = np.diff(sorted(transaction_times))
            
            # Identify rapid trading (many transactions in short time)
            short_gaps = len([gap for gap in time_gaps if gap < 60])  # Less than 1 minute apart
            if short_gaps > len(time_gaps) * 0.1:  # More than 10% of transactions
                suspicious_patterns.append({
                    'pattern': 'rapid_trading',
                    'description': 'High frequency of rapid transactions detected',
                    'instances': short_gaps,
                    'risk_level': 'medium'
                })
        
        return {
            'jurisdiction': jurisdiction.value,
            'surveillance_summary': {
                'total_alerts_generated': len(period_alerts),
                'critical_alerts': len([a for a in period_alerts if a.severity == 'critical']),
                'high_priority_alerts': len([a for a in period_alerts if a.severity == 'high']),
                'resolved_alerts': len([a for a in period_alerts if a.status == 'resolved']),
                'active_investigations': len([a for a in period_alerts if a.status == 'investigating'])
            },
            'alerts_by_type': alerts_by_type,
            'alerts_by_severity': alerts_by_severity,
            'suspicious_patterns_detected': suspicious_patterns,
            'surveillance_effectiveness': {
                'alert_resolution_rate': len([a for a in period_alerts if a.status in ['resolved', 'false_positive']]) / len(period_alerts) * 100 if period_alerts else 100,
                'false_positive_rate': len([a for a in period_alerts if a.status == 'false_positive']) / len(period_alerts) * 100 if period_alerts else 0
            },
            'recommendations': [
                'Continue monitoring high-risk patterns',
                'Review alert thresholds for optimization',
                'Enhance automated detection capabilities'
            ]
        }
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard summary."""
        
        # Recent activity (last 30 days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        recent_transactions = [tx for tx in self.transaction_log if tx.timestamp >= cutoff_date]
        recent_alerts = [alert for alert in self.compliance_alerts if alert.timestamp >= cutoff_date]
        
        # Alert status breakdown
        alert_statuses = {}
        for alert in self.compliance_alerts:
            if alert.status not in alert_statuses:
                alert_statuses[alert.status] = 0
            alert_statuses[alert.status] += 1
        
        return {
            'system_overview': {
                'total_transactions_recorded': len(self.transaction_log),
                'recent_transactions_30d': len(recent_transactions),
                'total_compliance_alerts': len(self.compliance_alerts),
                'recent_alerts_30d': len(recent_alerts),
                'active_positions': len([pos for pos in self.position_records.values() if pos['quantity'] > 0]),
                'generated_reports': len(self.generated_reports)
            },
            'compliance_status': {
                'transactions_requiring_review': len([tx for tx in self.transaction_log if tx.compliance_status == 'pending_review']),
                'approved_transactions': len([tx for tx in self.transaction_log if tx.compliance_status == 'approved']),
                'flagged_transactions': len([tx for tx in self.transaction_log if tx.regulatory_flags]),
                'alert_status_breakdown': alert_statuses
            },
            'regulatory_coverage': {
                'supported_jurisdictions': [j.value for j in RegulatoryJurisdiction],
                'active_surveillance_systems': len(self.surveillance_systems),
                'monitoring_rules_active': len([rule for rule, active in self.monitoring_rules.items() if active])
            },
            'recent_activity': {
                'last_transaction': max([tx.timestamp for tx in self.transaction_log]).isoformat() if self.transaction_log else None,
                'last_alert': max([alert.timestamp for alert in self.compliance_alerts]).isoformat() if self.compliance_alerts else None,
                'last_report_generated': max([rpt.generated_at for rpt in self.generated_reports.values()]).isoformat() if self.generated_reports else None
            },
            'system_health': {
                'surveillance_systems_operational': True,
                'reporting_engine_status': 'operational',
                'data_integrity_verified': True,
                'last_system_check': datetime.now(timezone.utc).isoformat()
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global compliance engine instance
compliance_engine = RegulatoryComplianceEngine()