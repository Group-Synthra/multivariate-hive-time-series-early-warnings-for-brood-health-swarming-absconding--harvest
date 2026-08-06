"""
=========================================================
Risk Level Classifier with Softmax Probability
=========================================================

Implements:
    P(swarming | x) = e^(z_swarm) / (e^(z_swarm) + e^(z_non-swarm))
    
Outputs:
    - Risk Percentage (0-100%)
    - Predicted Event Window (24-72 hours)
    - Risk Level (High/Medium/Low)
    - Key Contributing Factors
=========================================================
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskClassifier:
    """
    Advanced risk classifier with Softmax probability,
    event window estimation, and factor analysis.
    """
    
    def __init__(self):
        """Initialize classifier with risk thresholds."""
        
        # Risk thresholds (from config.py)
        self.RISK_LOW_MAX = 30
        self.RISK_MEDIUM_MAX = 60
        
        # Risk levels
        self.levels = ['LOW', 'MEDIUM', 'HIGH']
        
        # Event window mapping
        self.WINDOW_MAP = {
            'LOW': {
                'window': 'No immediate risk predicted',
                'hours': None,
                'confidence': 'High',
                'urgency': 'Routine'
            },
            'MEDIUM': {
                'window': 'Potential swarming within 48-72 hours',
                'hours': 60,
                'confidence': 'Medium',
                'urgency': 'Soon'
            },
            'HIGH': {
                'window': 'Swarming likely within 24-48 hours',
                'hours': 36,
                'confidence': 'High',
                'urgency': 'Immediate'
            }
        }
        
        # Risk messages
        self.RISK_MESSAGES = {
            'LOW': "Hive behaviour is normal. No immediate swarming risk detected.",
            'MEDIUM': "Possible behavioural changes detected. Continue monitoring hive.",
            'HIGH': "High swarming probability detected. Immediate hive inspection recommended."
        }
        
        # Feature names (must match your FEATURE_COLUMNS)
        self.feature_names = [
            'internal_temperature_c',
            'internal_humidity_pct',
            'co2_ppm',
            'hive_weight_kg',
            'external_temperature_c',
            'external_humidity_pct',
            'rainfall_mm_hour',
            'wind_speed_mps',
            'breakpoint',
            'days_since_breakpoint',
            'breakpoint_density',
            'segment_duration'
        ]
        
        # Normal ranges for sensor values
        self.normal_ranges = {
            'internal_temperature_c': (30, 36),
            'internal_humidity_pct': (50, 80),
            'co2_ppm': (400, 1500),
            'hive_weight_kg': (20, 50),
            'external_temperature_c': (15, 35),
            'external_humidity_pct': (40, 80),
            'battery_voltage': (3.0, 4.5),
            'breakpoint_density': (0, 1),
        }
    
    def softmax_binary(self, logits: float) -> float:
        """
        Binary Softmax for swarming probability.
        
        P(swarming | x) = e^(z_swarm) / (e^(z_swarm) + e^(z_non-swarm))
        
        Args:
            logits: Raw output from LSTM dense layer
        
        Returns:
            Probability of swarming (0-1)
        """
        # For binary classification with 2 classes
        z_swarm = logits
        z_non_swarm = 0  # Reference class
        
        # Softmax formula
        e_swarm = np.exp(z_swarm)
        e_non_swarm = np.exp(z_non_swarm)
        
        probability = e_swarm / (e_swarm + e_non_swarm)
        
        # Clamp to avoid numerical issues
        return float(np.clip(probability, 0.0001, 0.9999))
    
    def sigmoid_to_logits(self, sigmoid_output: float) -> float:
        """
        Convert sigmoid output to logits.
        
        logits = log(p / (1-p))
        """
        p = np.clip(sigmoid_output, 0.0001, 0.9999)
        return np.log(p / (1 - p))
    
    def classify_from_probability(self, probability: float) -> Dict:
        """
        Classify risk from LSTM probability.
        
        Args:
            probability: LSTM sigmoid output (0-1)
        
        Returns:
            dict: Complete risk classification
        """
        # Convert to logits if needed
        if 0 <= probability <= 1:
            # Assume sigmoid output
            logits = self.sigmoid_to_logits(probability)
        else:
            logits = probability
        
        # Apply Softmax
        risk_probability = self.softmax_binary(logits)
        
        # Calculate risk percentage
        risk_percentage = risk_probability * 100
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_percentage)
        
        # Get event window
        event_window = self._get_event_window(risk_level, risk_percentage)
        
        # Get message
        message = self.RISK_MESSAGES[risk_level]
        
        # Calculate softmax probabilities for each level
        softmax_probs = self._get_softmax_probabilities(risk_percentage)
        
        return {
            'risk_percentage': round(risk_percentage, 2),
            'risk_level': risk_level,
            'probability': round(risk_probability, 4),
            'event_window': event_window,
            'message': message,
            'softmax_probabilities': softmax_probs,
            'formula': 'P(swarming | x) = e^(z_swarm) / (e^(z_swarm) + e^(z_non-swarm))',
            'timestamp': datetime.now().isoformat()
        }
    
    def classify_with_factors(self, 
                              probability: float, 
                              sequence: Optional[np.ndarray] = None,
                              sensor_values: Optional[Dict] = None) -> Dict:
        """
        Classify risk with key factor identification.
        
        Args:
            probability: LSTM sigmoid output (0-1)
            sequence: Full sequence for factor analysis (optional)
            sensor_values: Latest sensor values (optional)
        
        Returns:
            dict: Complete risk classification with factors
        """
        # Get base classification
        result = self.classify_from_probability(probability)
        
        # Add key factors if data available
        if sensor_values:
            key_factors = self._identify_key_factors(sensor_values)
            result['key_factors'] = key_factors
        
        # Add recommendations
        result['recommendations'] = self._get_recommendations(
            result['risk_level'],
            result.get('key_factors', [])
        )
        
        return result
    
    def _get_risk_level(self, risk_percentage: float) -> str:
        """Map risk percentage to risk level."""
        if risk_percentage <= self.RISK_LOW_MAX:
            return 'LOW'
        elif risk_percentage <= self.RISK_MEDIUM_MAX:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _get_event_window(self, risk_level: str, risk_percentage: float) -> Dict:
        """
        Predict event window based on risk level.
        """
        base = self.WINDOW_MAP[risk_level].copy()
        
        if risk_level == 'MEDIUM':
            # 48-72 hours for medium risk
            hours = int(72 - (risk_percentage - 30) * 1.2)
            hours = max(48, min(72, hours))
            base['window'] = f'Potential swarming within {hours-12} to {hours+12} hours'
            base['hours'] = hours
            base['range'] = [hours-12, hours+12]
            
        elif risk_level == 'HIGH':
            # 24-48 hours for high risk
            hours = int(48 - (risk_percentage - 60) * 0.8)
            hours = max(24, min(48, hours))
            base['window'] = f'Swarming likely within {hours-6} to {hours+6} hours'
            base['hours'] = hours
            base['range'] = [hours-6, hours+6]
        
        return base
    
    def _get_softmax_probabilities(self, risk_percentage: float) -> Dict:
        """
        Get softmax probabilities for each risk level.
        
        This converts the single risk percentage into a probability
        distribution across LOW, MEDIUM, HIGH.
        """
        normalized_risk = risk_percentage / 100
        
        # Raw scores for each level
        low_score = max(0.01, 1 - normalized_risk * 1.5)
        medium_score = max(0.01, 1 - abs(normalized_risk - 0.5) * 2)
        high_score = max(0.01, normalized_risk * 1.5)
        
        # Apply softmax
        scores = np.array([low_score, medium_score, high_score])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        return {
            'LOW': round(probs[0] * 100, 2),
            'MEDIUM': round(probs[1] * 100, 2),
            'HIGH': round(probs[2] * 100, 2)
        }
    
    def get_risk_color(self, risk_level: str) -> str:
        """
        Get color for risk level.
        
        Args:
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'
        
        Returns:
            Hex color code
        """
        colors = {
            'LOW': '#22c55e',
            'MEDIUM': '#eab308',
            'HIGH': '#ef4444'
        }
        return colors.get(risk_level, '#94a3b8')
    
    def _identify_key_factors(self, sensor_values: Dict) -> List[Dict]:
        """
        Identify which sensors contributed most to the prediction.
        
        Based on deviation from normal ranges.
        """
        factors = []
        
        for key, value in sensor_values.items():
            if key in self.normal_ranges:
                low, high = self.normal_ranges[key]
                
                # Skip if value is None or invalid
                if value is None:
                    continue
                
                try:
                    val = float(value)
                except (ValueError, TypeError):
                    continue
                
                # Check if value is outside normal range
                if val < low:
                    deviation = ((low - val) / low) * 100
                    if deviation > 5:  # Only report significant deviations
                        factors.append({
                            'feature': key,
                            'value': round(val, 2),
                            'deviation': f'{deviation:.1f}% below normal',
                            'influence': 'Low' if deviation < 20 else 'Medium' if deviation < 50 else 'High'
                        })
                elif val > high:
                    deviation = ((val - high) / high) * 100
                    if deviation > 5:
                        factors.append({
                            'feature': key,
                            'value': round(val, 2),
                            'deviation': f'{deviation:.1f}% above normal',
                            'influence': 'Low' if deviation < 20 else 'Medium' if deviation < 50 else 'High'
                        })
        
        # Sort by influence (High → Medium → Low)
        influence_order = {'High': 0, 'Medium': 1, 'Low': 2}
        factors.sort(key=lambda x: influence_order.get(x['influence'], 3))
        
        return factors[:5]  # Top 5 factors
    
    def _get_recommendations(self, risk_level: str, factors: List) -> List[str]:
        """Get detailed recommendations based on risk level and factors."""
        recommendations = {
            'LOW': [
                '✓ Continue routine hive monitoring',
                '✓ Maintain normal hive ventilation',
                '✓ Ensure adequate food and water availability',
                '✓ Review sensor readings during the next upda'
            ],
            'MEDIUM': [
                '⚠️ Conduct hive inspection within 24-48 hours',
                '⚠️ Check for presence of queen cells',
                '⚠️ Monitor colony population and congestion',
                '⚠️ Consider providing additional space if needed',
                '⚠️ Review sensor data daily'
            ],
            'HIGH': [
                '🚨 Perform immediate hive inspection TODAY!',
                '🚨 Check ALL frames for queen cells',
                '🚨 Reduce overcrowding - consider hive splitting',
                '🚨 Prepare swarm traps if possible',
                '🚨 Monitor temperature and CO₂ levels closely',
                '🚨 Contact experienced beekeeper if needed'
            ]
        }
        
        base_recs = recommendations.get(risk_level, recommendations['LOW'])
        
        # Add factor-specific recommendations
        factor_recs = []
        for factor in factors[:2]:  # Top 2 factors
            feature = factor['feature']
            if 'temperature' in feature.lower():
                factor_recs.append(f'  → Temperature is {factor["deviation"]} - check ventilation')
            elif 'weight' in feature.lower():
                factor_recs.append(f'  → Hive weight is {factor["deviation"]} - check food stores')
            elif 'co2' in feature.lower():
                factor_recs.append(f'  → CO₂ is {factor["deviation"]} - check hive congestion')
            elif 'humidity' in feature.lower():
                factor_recs.append(f'  → Humidity is {factor["deviation"]} - check moisture levels')
        
        return base_recs + factor_recs
    
    def format_report(self, classification: Dict) -> str:
        """
        Format classification as a readable report.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SWARMING PREDICTION REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"📊 Risk Percentage: {classification['risk_percentage']}%")
        lines.append(f"📈 Risk Level: {classification['risk_level']}")
        
        if classification.get('event_window'):
            lines.append(f"🎯 Event Window: {classification['event_window']['window']}")
        
        lines.append("")
        
        if classification.get('softmax_probabilities'):
            lines.append("📊 Probability Distribution:")
            probs = classification['softmax_probabilities']
            lines.append(f"  • LOW: {probs['LOW']}%")
            lines.append(f"  • MEDIUM: {probs['MEDIUM']}%")
            lines.append(f"  • HIGH: {probs['HIGH']}%")
            lines.append("")
        
        if classification.get('key_factors'):
            lines.append("🔑 Key Factors:")
            for factor in classification['key_factors']:
                lines.append(f"  • {factor['feature']}: {factor['value']} ({factor['deviation']})")
            lines.append("")
        
        lines.append("📝 Recommendations:")
        if classification.get('recommendations'):
            for rec in classification['recommendations']:
                lines.append(f"  {rec}")
        else:
            lines.append(f"  {classification.get('message', 'No recommendations available')}")
        
        lines.append("")
        lines.append("📐 Formula Used:")
        lines.append(f"  {classification.get('formula', 'P(swarming | x) = e^(z_swarm) / (e^(z_swarm) + e^(z_non-swarm))')}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Factory function for easy integration
# ──────────────────────────────────────────────────────────────

def create_risk_classifier() -> RiskClassifier:
    """Create and return a RiskClassifier instance."""
    return RiskClassifier()


# ──────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create classifier
    classifier = RiskClassifier()
    
    # Test get_risk_color
    print("Risk Colors:")
    print(f"  LOW: {classifier.get_risk_color('LOW')}")
    print(f"  MEDIUM: {classifier.get_risk_color('MEDIUM')}")
    print(f"  HIGH: {classifier.get_risk_color('HIGH')}")
    
    # Example: Classify with different probabilities
    for prob in [0.02, 0.15, 0.45, 0.85]:
        result = classifier.classify_from_probability(prob)
        print(f"\nProbability: {prob:.2f} → {result['risk_level']} ({result['risk_percentage']:.1f}%)")
        print(f"  Event: {result['event_window']['window']}")
        print(f"  Distribution: {result['softmax_probabilities']}")