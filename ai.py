# ai_engine.py
"""
Advanced Naval AI Engine
Focused ML system for intelligent vessel generation, threat analysis, and decision making
"""

import numpy as np
import random
import math
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    NEUTRAL = "neutral"
    POSSIBLE = "possible"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"


class AIAction(Enum):
    INTERCEPT = "intercept"
    SAFE = "safe"
    MONITOR = "monitor"
    IGNORE = "ignore"


@dataclass
class VesselData:
    """Standardized vessel data structure for ML processing"""
    id: str
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    threat_level: ThreatLevel
    vessel_type: str
    crew_count: int
    items: List[str]
    weapons: List[str]
    scanned: bool = False


@dataclass
class AIReport:
    """Standardized AI decision report"""
    vessel_id: str
    threat_assessment: str
    recommended_action: AIAction
    confidence: float
    reasoning: str
    timestamp: float


class NavalAI:
    """
    Advanced Naval AI Engine
    Handles intelligent vessel generation, threat analysis, and decision support
    """

    def __init__(self, backend=None):
        self.backend = backend
        self.history = []
        self.vessel_templates = self._load_vessel_templates()
        self.threat_patterns = self._load_threat_patterns()
        
        # AI configuration
        self.generation_radius = 800
        self.safe_distance = 50
        self.analysis_interval = 2.0
        
        # Performance tracking
        self.performance_metrics = {
            "vessels_generated": 0,
            "threats_identified": 0,
            "decisions_made": 0,
            "avg_confidence": 0.0
        }

    def _load_vessel_templates(self) -> Dict:
        """Load intelligent vessel templates"""
        return {
            "civilian": {
                "types": ["Cargo Ship", "Fishing Vessel", "Merchant Ship", "Pleasure Craft"],
                "speed_range": (0.5, 3.0),
                "crew_range": (5, 25),
                "items": ["cargo", "supplies", "fishing_gear", "containers"],
                "weapons": [],
                "base_threat": ThreatLevel.NEUTRAL,
                "spawn_weight": 0.6
            },
            "military": {
                "types": ["Patrol Boat", "Coastal Defense", "Interceptor", "Gunboat"],
                "speed_range": (2.0, 6.0),
                "crew_range": (8, 30),
                "items": ["radar", "communication_gear", "sensors"],
                "weapons": ["machine_gun", "cannon", "missiles"],
                "base_threat": ThreatLevel.POSSIBLE,
                "spawn_weight": 0.3
            },
            "hostile": {
                "types": ["Attack Boat", "Raider", "Combat Ship", "Stealth Vessel"],
                "speed_range": (3.0, 8.0),
                "crew_range": (10, 40),
                "items": ["advanced_radar", "e_war_suite", "stealth_systems"],
                "weapons": ["heavy_machine_gun", "rockets", "torpedoes", "missiles"],
                "base_threat": ThreatLevel.CONFIRMED,
                "spawn_weight": 0.1
            }
        }

    def _load_threat_patterns(self) -> Dict:
        """Load threat detection patterns"""
        return {
            "suspicious_movement": {
                "description": "Erratic or aggressive movement",
                "indicators": ["high_speed", "zigzag_pattern", "direct_approach"],
                "threat_increase": 0.3
            },
            "weapon_signatures": {
                "description": "Detection of combat systems",
                "indicators": ["weapon_emissions", "armor_detection", "combat_systems"],
                "threat_increase": 0.6
            },
            "stealth_behavior": {
                "description": "Attempts to avoid detection",
                "indicators": ["radio_silence", "low_emissions", "covert_routing"],
                "threat_increase": 0.4
            }
        }

    def generate_vessels(self, count: int, mission_type: str = "patrol") -> List[Dict]:
        """
        Intelligently generate vessels with collision avoidance and mission-appropriate distribution
        """
        vessels = []
        existing_positions = self._get_existing_vessel_positions()
        
        # Adjust vessel distribution based on mission
        threat_weights = self._get_mission_threat_weights(mission_type)
        
        for i in range(count):
            # Select vessel type based on mission
            template_key = self._select_vessel_template(threat_weights)
            template = self.vessel_templates[template_key]
            
            # Generate safe position
            position = self._generate_safe_position(existing_positions)
            if position is None:
                continue
                
            existing_positions.append(position)
            
            # Generate vessel attributes
            velocity = self._generate_velocity(template["speed_range"])
            vessel_type = random.choice(template["types"])
            crew_count = random.randint(*template["crew_range"])
            
            # Determine final threat level (can be modified by behavior)
            final_threat = self._determine_final_threat(template["base_threat"])
            
            vessel_data = {
                "id": f"vessel_{len(vessels)}_{random.randint(1000,9999)}",
                "position": position,
                "velocity": velocity,
                "threat_level": final_threat.value,
                "vessel_type": vessel_type,
                "crew_count": crew_count,
                "items": template["items"].copy(),
                "weapons": template["weapons"].copy(),
                "scanned": False
            }
            
            # Add behavioral variations
            self._customize_vessel_behavior(vessel_data, template_key)
            
            vessels.append(vessel_data)
            self.performance_metrics["vessels_generated"] += 1
        
        return vessels

    def _get_mission_threat_weights(self, mission_type: str) -> Dict[str, float]:
        """Get vessel type weights based on mission"""
        if mission_type == "attack_vessel":
            return {"civilian": 0.2, "military": 0.4, "hostile": 0.4}
        elif mission_type == "patrol_boat":
            return {"civilian": 0.5, "military": 0.3, "hostile": 0.2}
        else:  # Default patrol
            return {"civilian": 0.6, "military": 0.3, "hostile": 0.1}

    def _select_vessel_template(self, weights: Dict[str, float]) -> str:
        """Select vessel template based on weights"""
        templates = list(weights.keys())
        template_weights = [weights[key] for key in templates]
        return random.choices(templates, weights=template_weights)[0]

    def _get_existing_vessel_positions(self) -> List[Tuple[float, float]]:
        """Get positions of existing vessels for collision avoidance"""
        try:
            if self.backend and hasattr(self.backend, 'units'):
                return [(unit.x, unit.y) for unit in self.backend.units if hasattr(unit, 'active') and unit.active]
        except:
            pass
        return []

    def _generate_safe_position(self, existing_positions: List[Tuple[float, float]], 
                              max_attempts: int = 50) -> Optional[Tuple[float, float]]:
        """Generate collision-free position using numpy for efficiency"""
        for attempt in range(max_attempts):
            # Prefer positions away from center for more interesting gameplay
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(200, self.generation_radius)
            x = 400 + distance * math.cos(angle)
            y = 300 + distance * math.sin(angle)
            
            position = (x, y)
            
            if self._is_position_safe(position, existing_positions):
                return position
        
        # Fallback: any position
        x = random.uniform(50, 750)
        y = random.uniform(50, 550)
        return (x, y)

    def _is_position_safe(self, position: Tuple[float, float], 
                         existing_positions: List[Tuple[float, float]]) -> bool:
        """Check if position is safe using numpy distance calculation"""
        if not existing_positions:
            return True
            
        pos_array = np.array(position)
        existing_array = np.array(existing_positions)
        
        distances = np.linalg.norm(existing_array - pos_array, axis=1)
        return np.all(distances > self.safe_distance)

    def _generate_velocity(self, speed_range: Tuple[float, float]) -> Tuple[float, float]:
        """Generate realistic velocity vector"""
        speed = random.uniform(*speed_range)
        # Bias movement toward center for more interaction
        center_x, center_y = 400, 300
        angle_to_center = math.atan2(center_y, center_x)
        angle_variation = random.uniform(-math.pi/4, math.pi/4)
        angle = angle_to_center + angle_variation
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        return (vx, vy)

    def _determine_final_threat(self, base_threat: ThreatLevel) -> ThreatLevel:
        """Add randomness to threat levels for more dynamic gameplay"""
        if base_threat == ThreatLevel.NEUTRAL:
            # 10% chance of being suspicious
            if random.random() < 0.1:
                return ThreatLevel.POSSIBLE
        elif base_threat == ThreatLevel.POSSIBLE:
            # 20% chance of being confirmed threat
            if random.random() < 0.2:
                return ThreatLevel.CONFIRMED
            # 10% chance of being neutral
            elif random.random() < 0.1:
                return ThreatLevel.NEUTRAL
        
        return base_threat

    def _customize_vessel_behavior(self, vessel_data: Dict, template_key: str):
        """Add behavioral variations to vessels"""
        # Military vessels might have hidden capabilities
        if template_key == "military" and random.random() < 0.3:
            vessel_data["items"].append("advanced_sensors")
            if random.random() < 0.2:
                vessel_data["weapons"].append("hidden_missiles")
        
        # Hostile vessels might pretend to be civilian
        if template_key == "hostile" and random.random() < 0.4:
            vessel_data["vessel_type"] = random.choice(["Cargo Ship", "Fishing Vessel"])
            vessel_data["items"].append("disguise_equipment")

    def analyze_environment(self, units: List) -> Dict:
        """
        Analyze current environment for threats and patterns
        Returns comprehensive threat assessment
        """
        try:
            analysis = {
                "timestamp": 0.0,
                "total_vessels": len(units) - 1,  # Exclude player
                "threat_summary": {
                    "neutral": 0,
                    "possible": 0,
                    "confirmed": 0,
                    "unknown": 0
                },
                "detected_patterns": [],
                "risk_assessment": "low",
                "recommendations": [],
                "ai_confidence": 0.8
            }
            
            # Analyze each vessel
            for unit in units[1:]:  # Skip player
                if not hasattr(unit, 'active') or not unit.active:
                    continue
                    
                threat_level = getattr(unit, 'true_threat_level', ThreatLevel.UNKNOWN)
                if hasattr(threat_level, 'value'):
                    threat_level = threat_level.value
                
                analysis["threat_summary"][threat_level] += 1
                
                # Detect suspicious behavior
                patterns = self._analyze_vessel_behavior(unit)
                analysis["detected_patterns"].extend(patterns)
            
            # Calculate overall risk
            analysis["risk_assessment"] = self._calculate_risk_level(analysis["threat_summary"])
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_tactical_recommendations(analysis)
            
            self.performance_metrics["threats_identified"] += analysis["threat_summary"]["confirmed"]
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "risk_assessment": "unknown",
                "recommendations": ["Continue with caution"]
            }

    def _analyze_vessel_behavior(self, vessel) -> List[Dict]:
        """Analyze individual vessel behavior for suspicious patterns"""
        patterns = []
        
        if not hasattr(vessel, 'speed') or not hasattr(vessel, 'vessel_type'):
            return patterns
        
        # High speed for civilian vessel
        if vessel.speed > 4.0 and "Cargo" in vessel.vessel_type:
            patterns.append({
                "type": "suspicious_movement",
                "vessel_id": getattr(vessel, 'id', 'unknown'),
                "description": f"High speed ({vessel.speed:.1f}) for civilian vessel",
                "confidence": 0.7
            })
        
        # Erratic movement pattern
        if hasattr(vessel, 'vx') and hasattr(vessel, 'vy'):
            movement_angle = math.atan2(vessel.vy, vessel.vx)
            if abs(movement_angle) > math.pi/2:  # Frequent direction changes
                patterns.append({
                    "type": "suspicious_movement",
                    "vessel_id": getattr(vessel, 'id', 'unknown'),
                    "description": "Erratic movement pattern detected",
                    "confidence": 0.6
                })
        
        return patterns

    def _calculate_risk_level(self, threat_summary: Dict) -> str:
        """Calculate overall risk level based on threat distribution"""
        total = sum(threat_summary.values())
        if total == 0:
            return "low"
            
        threat_score = (
            threat_summary["possible"] * 0.5 + 
            threat_summary["confirmed"] * 1.0
        ) / total
        
        if threat_score > 0.6:
            return "high"
        elif threat_score > 0.3:
            return "medium"
        else:
            return "low"

    def _generate_tactical_recommendations(self, analysis: Dict) -> List[str]:
        """Generate tactical recommendations based on analysis"""
        recommendations = []
        risk_level = analysis["risk_assessment"]
        
        if risk_level == "high":
            recommendations.extend([
                "Immediate threat interception required",
                "Maintain defensive posture",
                "Consider tactical withdrawal"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor suspicious vessels closely",
                "Maintain combat readiness",
                "Prepare for potential escalation"
            ])
        else:
            recommendations.extend([
                "Continue routine patrol operations",
                "Verify identification of unknown vessels",
                "Maintain situational awareness"
            ])
        
        # Add specific recommendations based on patterns
        for pattern in analysis["detected_patterns"]:
            if pattern["confidence"] > 0.7:
                recommendations.append(f"High confidence pattern: {pattern['description']}")
        
        return recommendations

    def generate_status_report(self, vessel_id: str) -> Dict:
        """
        Generate AI status report for specific vessel
        Used by UI to display AI recommendations
        """
        try:
            # Simulate AI analysis (in real implementation, this would use actual vessel data)
            vessel_analysis = self._analyze_single_vessel(vessel_id)
            
            report = {
                "vessel_id": vessel_id,
                "threat_assessment": vessel_analysis["threat_level"],
                "recommended_action": vessel_analysis["recommended_action"].value,
                "confidence": vessel_analysis["confidence"],
                "reasoning": vessel_analysis["reasoning"],
                "timestamp": 0.0,
                "report_id": f"ai_report_{vessel_id}_{random.randint(1000,9999)}"
            }
            
            self.performance_metrics["decisions_made"] += 1
            self.history.append(report)
            
            return report
            
        except Exception as e:
            return self._create_error_report(vessel_id, str(e))

    def _analyze_single_vessel(self, vessel_id: str) -> Dict:
        """Analyze single vessel and make recommendation"""
        # This would normally analyze actual vessel data
        # For now, we'll simulate based on vessel ID pattern
        
        threat_levels = ["low", "medium", "high"]
        actions = [AIAction.INTERCEPT, AIAction.MONITOR, AIAction.SAFE]
        
        # Simple heuristic based on vessel ID
        if "hostile" in vessel_id:
            threat = "high"
            action = AIAction.INTERCEPT
            confidence = 0.9
            reasoning = "Vessel matches hostile patterns - immediate interception recommended"
        elif "military" in vessel_id:
            threat = "medium"
            action = AIAction.MONITOR
            confidence = 0.7
            reasoning = "Military vessel detected - monitor for suspicious activity"
        else:
            threat = "low"
            action = AIAction.SAFE
            confidence = 0.8
            reasoning = "Civilian vessel pattern - likely safe but verify identification"
        
        return {
            "threat_level": threat,
            "recommended_action": action,
            "confidence": confidence,
            "reasoning": reasoning
        }

    def _create_error_report(self, vessel_id: str, error: str) -> Dict:
        """Create error report when analysis fails"""
        return {
            "vessel_id": vessel_id,
            "threat_assessment": "unknown",
            "recommended_action": AIAction.MONITOR.value,
            "confidence": 0.1,
            "reasoning": f"Analysis error: {error}",
            "timestamp": 0.0,
            "report_id": f"error_report_{vessel_id}"
        }

    def update_with_performance(self, performance_data: Dict):
        """Update AI with performance data for learning"""
        if not performance_data:
            return
            
        # Update confidence metrics
        if "accuracy" in performance_data:
            new_confidence = performance_data["accuracy"]
            self.performance_metrics["avg_confidence"] = (
                0.8 * self.performance_metrics["avg_confidence"] + 
                0.2 * new_confidence
            )

    def get_ai_insights(self, game_state: Dict) -> List[str]:
        """Generate real-time AI insights for UI display"""
        insights = []
        
        threats_remaining = game_state.get("threats_remaining", 0)
        player_accuracy = game_state.get("accuracy", 0.5)
        
        # Threat-based insights
        if threats_remaining > 5:
            insights.append("🚨 High threat concentration - prioritize targets")
        elif threats_remaining > 2:
            insights.append("⚠️ Multiple threats detected - maintain awareness")
        else:
            insights.append("✅ Threat level manageable - continue patrol")
        
        # Performance-based insights
        if player_accuracy < 0.3:
            insights.append("🎯 Low accuracy detected - verify targets carefully")
        elif player_accuracy > 0.8:
            insights.append("🎯 Excellent accuracy - maintain focus")
        
        # Add tactical advice
        tactical_advice = [
            "Scan sectors systematically",
            "Maintain safe distance from unknowns",
            "Watch for flanking maneuvers",
            "Conserve intercepts for confirmed threats"
        ]
        insights.append(random.choice(tactical_advice))
        
        return insights

    def get_performance_stats(self) -> Dict:
        """Get AI performance statistics"""
        return {
            "vessels_generated": self.performance_metrics["vessels_generated"],
            "threats_identified": self.performance_metrics["threats_identified"],
            "decisions_made": self.performance_metrics["decisions_made"],
            "average_confidence": self.performance_metrics["avg_confidence"],
            "total_reports": len(self.history)
        }


# Factory function for easy integration
def create_naval_ai(backend=None) -> NavalAI:
    """Create and initialize Naval AI system"""
    return NavalAI(backend)


# Utility functions
def calculate_interception_point(interceptor_pos: Tuple[float, float], 
                               interceptor_speed: float,
                               target_pos: Tuple[float, float],
                               target_velocity: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate optimal interception point"""
    interceptor_pos = np.array(interceptor_pos)
    target_pos = np.array(target_pos)
    target_vel = np.array(target_velocity)
    
    # Simplified interception calculation
    rel_pos = target_pos - interceptor_pos
    distance = np.linalg.norm(rel_pos)
    
    if distance == 0:
        return tuple(target_pos)
    
    # Estimate interception point
    time_to_intercept = distance / (interceptor_speed + 0.1)  # Small buffer
    interception_point = target_pos + target_vel * time_to_intercept
    
    return tuple(interception_point)


def validate_vessel_data(vessel_data: Dict) -> bool:
    """Validate vessel data structure"""
    required = ["id", "position", "velocity", "threat_level", "vessel_type"]
    return all(field in vessel_data for field in required)