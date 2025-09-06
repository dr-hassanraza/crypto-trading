"""
Validation Models for Crypto Trend Analyzer

Independent validation layer that acts as guardrails for trading decisions.
These models validate signals, risk levels, and market conditions before execution.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logging_config import crypto_logger
from src.core.error_handling_engine import handle_error_async, ErrorCategory


class ValidationResult(Enum):
    """Validation result types."""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    REQUIRES_REVIEW = "requires_review"


class ValidationCategory(Enum):
    """Types of validations."""
    SIGNAL_QUALITY = "signal_quality"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_CONDITIONS = "market_conditions"
    PORTFOLIO_LIMITS = "portfolio_limits"
    EXECUTION_TIMING = "execution_timing"
    DATA_QUALITY = "data_quality"


@dataclass
class ValidationReport:
    """Validation report for a trading decision."""
    validation_id: str
    timestamp: datetime
    category: ValidationCategory
    result: ValidationResult
    confidence_score: float
    reasons: List[str]
    recommendations: List[str]
    risk_score: float
    metadata: Dict[str, Any]


class SignalQualityValidator:
    """Validates trading signal quality and consistency."""
    
    def __init__(self):
        self.min_confidence_threshold = 0.65
        self.signal_history: List[Dict] = []
        self.consistency_window = 5
    
    async def validate_signal(self, signal_data: Dict[str, Any]) -> ValidationReport:
        """Validate signal quality."""
        try:
            reasons = []
            recommendations = []
            confidence = signal_data.get('final_signal', {}).get('confidence_score', 0) / 100
            
            # Confidence threshold check
            if confidence < self.min_confidence_threshold:
                return ValidationReport(
                    validation_id=f"sig_val_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    timestamp=datetime.now(),
                    category=ValidationCategory.SIGNAL_QUALITY,
                    result=ValidationResult.REJECTED,
                    confidence_score=confidence,
                    reasons=[f"Signal confidence {confidence:.2f} below threshold {self.min_confidence_threshold}"],
                    recommendations=["Wait for higher confidence signal", "Review model parameters"],
                    risk_score=0.8,
                    metadata=signal_data
                )
            
            # Signal consistency check
            consistency_score = await self._check_signal_consistency(signal_data)
            if consistency_score < 0.5:
                reasons.append(f"Low signal consistency: {consistency_score:.2f}")
                recommendations.append("Consider signal averaging")
            
            # Technical indicator alignment
            alignment_score = await self._check_indicator_alignment(signal_data)
            if alignment_score < 0.6:
                reasons.append(f"Poor indicator alignment: {alignment_score:.2f}")
                recommendations.append("Review technical indicators")
            
            # Determine final result
            overall_score = (confidence + consistency_score + alignment_score) / 3
            
            if overall_score >= 0.75:
                result = ValidationResult.APPROVED
                risk_score = 0.2
            elif overall_score >= 0.6:
                result = ValidationResult.CONDITIONAL
                risk_score = 0.5
                recommendations.append("Consider reduced position size")
            else:
                result = ValidationResult.REJECTED
                risk_score = 0.8
                recommendations.append("Wait for better signal quality")
            
            # Store signal for history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signal': signal_data.get('final_signal', {}).get('signal'),
                'confidence': confidence,
                'symbol': signal_data.get('symbol')
            })
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.signal_history = [
                s for s in self.signal_history 
                if s['timestamp'] > cutoff_time
            ]
            
            return ValidationReport(
                validation_id=f"sig_val_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.SIGNAL_QUALITY,
                result=result,
                confidence_score=overall_score,
                reasons=reasons,
                recommendations=recommendations,
                risk_score=risk_score,
                metadata={'signal_data': signal_data, 'scores': {
                    'confidence': confidence,
                    'consistency': consistency_score,
                    'alignment': alignment_score
                }}
            )
            
        except Exception as e:
            await handle_error_async(e, {'component': 'signal_quality_validator'})
            return ValidationReport(
                validation_id=f"sig_val_err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.SIGNAL_QUALITY,
                result=ValidationResult.REJECTED,
                confidence_score=0.0,
                reasons=["Validation error occurred"],
                recommendations=["Manual review required"],
                risk_score=1.0,
                metadata={'error': str(e)}
            )
    
    async def _check_signal_consistency(self, signal_data: Dict[str, Any]) -> float:
        """Check signal consistency with recent signals."""
        if len(self.signal_history) < 2:
            return 1.0  # No history to compare against
        
        current_signal = signal_data.get('final_signal', {}).get('signal')
        symbol = signal_data.get('symbol')
        
        # Get recent signals for same symbol
        recent_signals = [
            s for s in self.signal_history[-self.consistency_window:]
            if s['symbol'] == symbol
        ]
        
        if not recent_signals:
            return 1.0
        
        # Check for signal flipping (conflicting signals)
        signal_changes = 0
        for i in range(1, len(recent_signals)):
            if recent_signals[i]['signal'] != recent_signals[i-1]['signal']:
                signal_changes += 1
        
        # Consistency score (lower changes = higher consistency)
        consistency_score = max(0.0, 1.0 - (signal_changes / len(recent_signals)))
        return consistency_score
    
    async def _check_indicator_alignment(self, signal_data: Dict[str, Any]) -> float:
        """Check alignment between different technical indicators."""
        indicators = signal_data.get('technical_analysis', {})
        
        if not indicators:
            return 0.5  # Neutral if no indicators
        
        # Count bullish vs bearish indicators
        bullish_count = 0
        bearish_count = 0
        total_indicators = 0
        
        for indicator, data in indicators.items():
            if isinstance(data, dict) and 'signal' in data:
                total_indicators += 1
                signal = data['signal'].upper()
                if 'BUY' in signal or 'BULL' in signal:
                    bullish_count += 1
                elif 'SELL' in signal or 'BEAR' in signal:
                    bearish_count += 1
        
        if total_indicators == 0:
            return 0.5
        
        # Alignment score (higher when indicators agree)
        alignment_ratio = max(bullish_count, bearish_count) / total_indicators
        return alignment_ratio


class RiskAssessmentValidator:
    """Validates risk levels for trading decisions."""
    
    def __init__(self):
        self.max_portfolio_risk = 0.15  # 15% max portfolio at risk
        self.max_position_risk = 0.05   # 5% max single position risk
        self.volatility_threshold = 0.08  # 8% daily volatility threshold
    
    async def validate_risk(self, signal_data: Dict[str, Any], 
                          portfolio_data: Dict[str, Any]) -> ValidationReport:
        """Validate risk levels for a potential trade."""
        try:
            reasons = []
            recommendations = []
            risk_checks = []
            
            # Position size risk check
            position_risk = await self._calculate_position_risk(signal_data, portfolio_data)
            risk_checks.append(position_risk)
            
            if position_risk > self.max_position_risk:
                reasons.append(f"Position risk {position_risk:.1%} exceeds limit {self.max_position_risk:.1%}")
                recommendations.append("Reduce position size")
            
            # Portfolio risk check
            portfolio_risk = await self._calculate_portfolio_risk(portfolio_data)
            risk_checks.append(portfolio_risk)
            
            if portfolio_risk > self.max_portfolio_risk:
                reasons.append(f"Portfolio risk {portfolio_risk:.1%} exceeds limit {self.max_portfolio_risk:.1%}")
                recommendations.append("Close some positions before opening new ones")
            
            # Volatility check
            volatility_risk = await self._assess_volatility_risk(signal_data)
            risk_checks.append(volatility_risk)
            
            if volatility_risk > self.volatility_threshold:
                reasons.append(f"High volatility detected: {volatility_risk:.1%}")
                recommendations.append("Consider smaller position size")
            
            # Correlation risk (if multiple positions)
            correlation_risk = await self._assess_correlation_risk(signal_data, portfolio_data)
            risk_checks.append(correlation_risk)
            
            # Overall risk assessment
            overall_risk = np.mean(risk_checks)
            
            if overall_risk <= 0.3:
                result = ValidationResult.APPROVED
                risk_score = overall_risk
            elif overall_risk <= 0.6:
                result = ValidationResult.CONDITIONAL
                risk_score = overall_risk
                recommendations.append("Proceed with caution")
            else:
                result = ValidationResult.REJECTED
                risk_score = overall_risk
                recommendations.append("Risk too high - do not trade")
            
            return ValidationReport(
                validation_id=f"risk_val_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.RISK_ASSESSMENT,
                result=result,
                confidence_score=1.0 - overall_risk,
                reasons=reasons,
                recommendations=recommendations,
                risk_score=risk_score,
                metadata={
                    'position_risk': position_risk,
                    'portfolio_risk': portfolio_risk,
                    'volatility_risk': volatility_risk,
                    'correlation_risk': correlation_risk
                }
            )
            
        except Exception as e:
            await handle_error_async(e, {'component': 'risk_assessment_validator'})
            return ValidationReport(
                validation_id=f"risk_val_err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.RISK_ASSESSMENT,
                result=ValidationResult.REJECTED,
                confidence_score=0.0,
                reasons=["Risk validation error occurred"],
                recommendations=["Manual risk review required"],
                risk_score=1.0,
                metadata={'error': str(e)}
            )
    
    async def _calculate_position_risk(self, signal_data: Dict[str, Any], 
                                     portfolio_data: Dict[str, Any]) -> float:
        """Calculate risk for a single position."""
        entry_price = signal_data.get('final_signal', {}).get('entry_price', 0)
        stop_loss = signal_data.get('final_signal', {}).get('stop_loss', 0)
        portfolio_value = portfolio_data.get('total_value', 1)
        
        if not all([entry_price, stop_loss, portfolio_value]):
            return 0.5  # Default medium risk if data missing
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        # Assume 10% of portfolio for position sizing
        position_value_ratio = 0.1
        position_risk = risk_per_unit * position_value_ratio
        
        return min(position_risk, 1.0)
    
    async def _calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk."""
        positions = portfolio_data.get('positions', {})
        total_value = portfolio_data.get('total_value', 1)
        
        total_risk = 0
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            # Estimate risk as percentage of position value
            position_risk = (position_value / total_value) * 0.2  # Assume 20% risk per position
            total_risk += position_risk
        
        return min(total_risk, 1.0)
    
    async def _assess_volatility_risk(self, signal_data: Dict[str, Any]) -> float:
        """Assess volatility risk."""
        market_data = signal_data.get('market_data', {})
        volatility = market_data.get('volatility', 0.05)  # Default 5% volatility
        
        # Normalize volatility to 0-1 scale
        return min(volatility / 0.2, 1.0)  # 20% volatility = max risk
    
    async def _assess_correlation_risk(self, signal_data: Dict[str, Any], 
                                     portfolio_data: Dict[str, Any]) -> float:
        """Assess correlation risk with existing positions."""
        # Simplified correlation risk assessment
        symbol = signal_data.get('symbol', '')
        positions = portfolio_data.get('positions', {})
        
        # If no existing positions, no correlation risk
        if not positions:
            return 0.0
        
        # Simple heuristic: crypto positions are highly correlated
        crypto_positions = len(positions)
        if crypto_positions > 3:
            return 0.6  # High correlation risk
        elif crypto_positions > 1:
            return 0.3  # Medium correlation risk
        
        return 0.1  # Low correlation risk


class MarketConditionsValidator:
    """Validates market conditions for trading."""
    
    def __init__(self):
        self.min_volume_threshold = 1000000  # Minimum daily volume
        self.max_spread_threshold = 0.005    # Maximum 0.5% spread
        self.market_hours_check = True
    
    async def validate_market_conditions(self, signal_data: Dict[str, Any]) -> ValidationReport:
        """Validate current market conditions."""
        try:
            reasons = []
            recommendations = []
            condition_scores = []
            
            # Liquidity check
            liquidity_score = await self._check_liquidity(signal_data)
            condition_scores.append(liquidity_score)
            
            if liquidity_score < 0.5:
                reasons.append("Low market liquidity detected")
                recommendations.append("Consider waiting for better liquidity")
            
            # Spread check
            spread_score = await self._check_spread(signal_data)
            condition_scores.append(spread_score)
            
            if spread_score < 0.5:
                reasons.append("High bid-ask spread detected")
                recommendations.append("Use limit orders to control slippage")
            
            # Market volatility check
            volatility_score = await self._check_market_volatility(signal_data)
            condition_scores.append(volatility_score)
            
            if volatility_score < 0.5:
                reasons.append("Extreme market volatility detected")
                recommendations.append("Reduce position sizes")
            
            # Overall market conditions
            overall_score = np.mean(condition_scores)
            
            if overall_score >= 0.7:
                result = ValidationResult.APPROVED
            elif overall_score >= 0.5:
                result = ValidationResult.CONDITIONAL
                recommendations.append("Proceed with extra caution")
            else:
                result = ValidationResult.REJECTED
                recommendations.append("Market conditions unfavorable")
            
            return ValidationReport(
                validation_id=f"market_val_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.MARKET_CONDITIONS,
                result=result,
                confidence_score=overall_score,
                reasons=reasons,
                recommendations=recommendations,
                risk_score=1.0 - overall_score,
                metadata={
                    'liquidity_score': liquidity_score,
                    'spread_score': spread_score,
                    'volatility_score': volatility_score
                }
            )
            
        except Exception as e:
            await handle_error_async(e, {'component': 'market_conditions_validator'})
            return ValidationReport(
                validation_id=f"market_val_err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                category=ValidationCategory.MARKET_CONDITIONS,
                result=ValidationResult.REJECTED,
                confidence_score=0.0,
                reasons=["Market conditions validation error"],
                recommendations=["Manual market review required"],
                risk_score=1.0,
                metadata={'error': str(e)}
            )
    
    async def _check_liquidity(self, signal_data: Dict[str, Any]) -> float:
        """Check market liquidity."""
        market_data = signal_data.get('market_data', {})
        volume = market_data.get('volume_24h', 0)
        
        # Score based on volume threshold
        if volume >= self.min_volume_threshold * 2:
            return 1.0  # Excellent liquidity
        elif volume >= self.min_volume_threshold:
            return 0.8  # Good liquidity
        elif volume >= self.min_volume_threshold * 0.5:
            return 0.6  # Adequate liquidity
        else:
            return 0.2  # Poor liquidity
    
    async def _check_spread(self, signal_data: Dict[str, Any]) -> float:
        """Check bid-ask spread."""
        market_data = signal_data.get('market_data', {})
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        
        if bid == 0 or ask == 0:
            return 0.5  # Unknown spread
        
        spread = (ask - bid) / ((ask + bid) / 2)
        
        if spread <= self.max_spread_threshold * 0.5:
            return 1.0  # Excellent spread
        elif spread <= self.max_spread_threshold:
            return 0.8  # Good spread
        elif spread <= self.max_spread_threshold * 2:
            return 0.5  # Acceptable spread
        else:
            return 0.2  # Poor spread
    
    async def _check_market_volatility(self, signal_data: Dict[str, Any]) -> float:
        """Check market volatility levels."""
        market_data = signal_data.get('market_data', {})
        volatility = market_data.get('volatility', 0.05)
        
        # Score based on volatility (lower volatility = better conditions)
        if volatility <= 0.02:
            return 1.0  # Low volatility
        elif volatility <= 0.05:
            return 0.8  # Normal volatility
        elif volatility <= 0.1:
            return 0.5  # High volatility
        else:
            return 0.2  # Extreme volatility


class ValidationEngine:
    """Main validation engine that coordinates all validators."""
    
    def __init__(self):
        self.signal_validator = SignalQualityValidator()
        self.risk_validator = RiskAssessmentValidator()
        self.market_validator = MarketConditionsValidator()
        self.validation_history: List[ValidationReport] = []
    
    async def comprehensive_validation(self, signal_data: Dict[str, Any], 
                                     portfolio_data: Dict[str, Any]) -> List[ValidationReport]:
        """Run comprehensive validation on trading decision."""
        try:
            validation_reports = []
            
            # Signal quality validation
            signal_report = await self.signal_validator.validate_signal(signal_data)
            validation_reports.append(signal_report)
            
            # Risk assessment validation
            risk_report = await self.risk_validator.validate_risk(signal_data, portfolio_data)
            validation_reports.append(risk_report)
            
            # Market conditions validation
            market_report = await self.market_validator.validate_market_conditions(signal_data)
            validation_reports.append(market_report)
            
            # Store validation history
            self.validation_history.extend(validation_reports)
            
            # Keep only recent validation history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.validation_history = [
                v for v in self.validation_history 
                if v.timestamp > cutoff_time
            ]
            
            return validation_reports
            
        except Exception as e:
            await handle_error_async(e, {'component': 'validation_engine'})
            return []
    
    def get_final_validation_decision(self, validation_reports: List[ValidationReport]) -> Tuple[ValidationResult, float, List[str]]:
        """Get final validation decision from all reports."""
        if not validation_reports:
            return ValidationResult.REJECTED, 0.0, ["No validation reports available"]
        
        # Check for any rejections
        rejections = [r for r in validation_reports if r.result == ValidationResult.REJECTED]
        if rejections:
            reasons = []
            for report in rejections:
                reasons.extend(report.reasons)
            return ValidationResult.REJECTED, 0.0, reasons
        
        # Check for conditional approvals
        conditionals = [r for r in validation_reports if r.result == ValidationResult.CONDITIONAL]
        if conditionals:
            reasons = []
            for report in validation_reports:
                reasons.extend(report.recommendations)
            avg_confidence = np.mean([r.confidence_score for r in validation_reports])
            return ValidationResult.CONDITIONAL, avg_confidence, reasons
        
        # All approved
        avg_confidence = np.mean([r.confidence_score for r in validation_reports])
        recommendations = []
        for report in validation_reports:
            recommendations.extend(report.recommendations)
        
        return ValidationResult.APPROVED, avg_confidence, recommendations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        if not self.validation_history:
            return {}
        
        recent_validations = self.validation_history
        
        stats = {
            "total_validations": len(recent_validations),
            "approval_rate": len([v for v in recent_validations if v.result == ValidationResult.APPROVED]) / len(recent_validations),
            "rejection_rate": len([v for v in recent_validations if v.result == ValidationResult.REJECTED]) / len(recent_validations),
            "average_confidence": np.mean([v.confidence_score for v in recent_validations]),
            "average_risk_score": np.mean([v.risk_score for v in recent_validations]),
            "validations_by_category": {}
        }
        
        # Group by category
        for validation in recent_validations:
            category = validation.category.value
            if category not in stats["validations_by_category"]:
                stats["validations_by_category"][category] = {
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "conditional": 0
                }
            
            stats["validations_by_category"][category]["total"] += 1
            stats["validations_by_category"][category][validation.result.value] += 1
        
        return stats


# Global validation engine instance
validation_engine = ValidationEngine()