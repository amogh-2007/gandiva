# ai.py
"""
Naval Combat AI Engine
Handles intelligent vessel generation, threat analysis, and decision making
Completely separate from UI - communicates only through backend
"""

import numpy as np
import random
import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class ThreatLevel(Enum):
    NEUTRAL = "neutral"
    POSSIBLE = "possible"
    CONFIRMED = "confirmed"


class AIAction(Enum):
    INTERCEPT = "intercept"
    SAFE = "safe"
    MONITOR = "monitor"
    IGNORE = "ignore"


@dataclass
class VesselData:
    """Data structure for vessel information"""
    id: str
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    threat_level: ThreatLevel
    vessel_type: str
    crew_count: int
    items: List[str]
    weapons: List[str]
    last_seen_time: float


@dataclass
class AIReport:
    """Standardized AI status report"""
    vessel_id: str
    threat_assessment: str
    recommended_action: AIAction
    confidence: float
    reasoning: str
    timestamp: float


class NavalAI:
    """
    Main AI Engine for Naval Combat Simulation
    Generates vessels intelligently, analyzes threats, and makes decisions
    """

    def __init__(self, backend):
        self.backend = backend  # Reference to simulation backend
        self.history = []  # Stores AI decisions for learning
        self.vessel_templates = self._load_vessel_templates()
        self.threat_patterns = self._load_threat_patterns()

        # AI configuration
        self.generation_radius = 800  # Area for vessel generation
        self.safe_distance = 50  # Minimum distance between vessels
        self.analysis_interval = 2.0  # Seconds between environment analysis

    def _load_vessel_templates(self) -> Dict:
        """Load vessel templates for intelligent generation"""
        return {
            "cargo_ship": {
                "types": ["Container Ship", "Oil Tanker", "Bulk Carrier"],
                "speed_range": (0.5, 2.0),
                "crew_range": (10, 30),
                "items": ["cargo", "supplies", "containers"],
                "weapons": [],
                "base_threat": ThreatLevel.NEUTRAL
            },
            "fishing_vessel": {
                "types": ["Fishing Boat", "Trawler", "Factory Ship"],
                "speed_range": (1.0, 3.0),
                "crew_range": (5, 15),
                "items": ["fishing_gear", "catch", "nets"],
                "weapons": [],
                "base_threat": ThreatLevel.NEUTRAL
            },
            "patrol_boat": {
                "types": ["Coastal Patrol", "Interceptor", "Gunboat"],
                "speed_range": (3.0, 6.0),
                "crew_range": (8, 20),
                "items": ["radar", "communication_gear"],
                "weapons": ["machine_gun", "cannon", "missiles"],
                "base_threat": ThreatLevel.POSSIBLE
            },
            "hostile_vessel": {
                "types": ["Attack Boat", "Raider", "Combat Ship"],
                "speed_range": (4.0, 8.0),
                "crew_range": (15, 40),
                "items": ["advanced_radar", "e_war_suite"],
                "weapons": ["heavy_machine_gun", "rockets", "torpedoes", "missiles"],
                "base_threat": ThreatLevel.CONFIRMED
            }
        }

    def _load_threat_patterns(self) -> Dict:
        """Load patterns for threat detection"""
        return {
            "suspicious_movement": {
                "description": "Erratic or aggressive movement patterns",
                "indicators": ["high_speed", "zigzag_pattern", "direct_approach"],
                "threat_increase": 0.3
            },
            "weapon_signatures": {
                "description": "Detection of weapons or combat systems",
                "indicators": ["radar_emissions", "weapon_signatures", "armor"],
                "threat_increase": 0.5
            },
            "stealth_behavior": {
                "description": "Attempts to avoid detection",
                "indicators": ["radio_silence", "low_emissions", "covert_routing"],
                "threat_increase": 0.4
            },
            "group_coordination": {
                "description": "Coordinated movement with other vessels",
                "indicators": ["formation_flying", "synchronized_moves", "relaying"],
                "threat_increase": 0.6
            }
        }

    def generate_vessels(self, count: int, mission_type: str = "patrol") -> List[Dict]:
        """
        Intelligently generate new vessels with collision avoidance
        Uses numpy for coordinate control and threat distribution
        """
        vessels = []
        existing_positions = self._get_existing_vessel_positions()

        for i in range(count):
            # Determine vessel type based on mission
            template_key = self._select_vessel_template(mission_type)
            template = self.vessel_templates[template_key]

            # Generate safe position
            position = self._generate_safe_position(existing_positions)
            if position is None:
                continue  # Skip if no safe position found

            existing_positions.append(position)

            # Generate velocity and other attributes
            velocity = self._generate_velocity(template["speed_range"])
            vessel_type = random.choice(template["types"])
            crew_count = random.randint(*template["crew_range"])

            # Create vessel data
            vessel_data = {
                "id": f"vessel_{len(existing_positions)}_{random.randint(1000,9999)}",
                "position": position,
                "velocity": velocity,
                "threat_level": template["base_threat"].value,
                "vessel_type": vessel_type,
                "crew_count": crew_count,
                "items": template["items"].copy(),
                "weapons": template["weapons"].copy(),
                "last_seen_time": time.time()
            }

            # Add random equipment variations
            self._customize_vessel(vessel_data, template_key)

            vessels.append(vessel_data)

        return vessels

    def _select_vessel_template(self, mission_type: str) -> str:
        """Select appropriate vessel template based on mission"""
        if mission_type == "combat":
            weights = [0.1, 0.2, 0.3, 0.4]  # More hostile vessels
        elif mission_type == "search":
            weights = [0.3, 0.3, 0.2, 0.2]  # Balanced mix
        else:  # patrol
            weights = [0.4, 0.3, 0.2, 0.1]  # Mostly civilian

        templates = list(self.vessel_templates.keys())
        return random.choices(templates, weights=weights)[0]

    def _get_existing_vessel_positions(self) -> List[Tuple[float, float]]:
        """Get positions of all existing vessels for collision avoidance"""
        try:
            boats = self.backend.get_all_boats()
            return [boat["position"] for boat in boats.values()]
        except Exception as e:
            print(f"Error getting vessel positions from backend: {e}")
            return []

    def _generate_safe_position(self, existing_positions: List[Tuple[float, float]],
                              max_attempts: int = 50) -> Optional[Tuple[float, float]]:
        """
        Generate a position that doesn't collide with existing vessels
        Uses numpy for efficient distance calculations
        """
        for attempt in range(max_attempts):
            # Generate random position within radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(100, self.generation_radius)
            x = 400 + distance * math.cos(angle)  # Center at (400, 300)
            y = 300 + distance * math.sin(angle)

            position = (x, y)

            # Check if position is safe
            if self._is_position_safe(position, existing_positions):
                return position

        return None  # No safe position found

    def _is_position_safe(self, position: Tuple[float, float],
                         existing_positions: List[Tuple[float, float]]) -> bool:
        """Check if position is safe from collisions"""
        if not existing_positions:
            return True

        pos_array = np.array(position)
        existing_array = np.array(existing_positions)

        distances = np.linalg.norm(existing_array - pos_array, axis=1)
        return np.all(distances > self.safe_distance)

    def _generate_velocity(self, speed_range: Tuple[float, float]) -> Tuple[float, float]:
        """Generate velocity vector with random direction"""
        speed = random.uniform(*speed_range)
        angle = random.uniform(0, 2 * math.pi)

        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        return (vx, vy)

    def _customize_vessel(self, vessel_data: Dict, template_key: str):
        """Add random variations to vessel equipment and capabilities"""
        # Add random items based on vessel type
        if template_key == "cargo_ship":
            extra_items = ["navigation_computer", "crane", "life_rafts"]
            vessel_data["items"].extend(random.sample(extra_items, random.randint(1, 2)))

        elif template_key == "patrol_boat":
            extra_weapons = ["flares", "sonar", "depth_charges"]
            if random.random() < 0.3:
                vessel_data["weapons"].extend(random.sample(extra_weapons, 1))

        elif template_key == "hostile_vessel":
            # Hostile vessels might have hidden weapons
            hidden_weapons = ["stealth_system", "electronic_warfare", "decoy_launchers"]
            if random.random() < 0.5:
                vessel_data["items"].extend(random.sample(hidden_weapons, 1))

    def analyze_environment(self) -> Dict:
        """
        Analyze current environment for threats and anomalies
        Returns comprehensive threat assessment
        """
        try:
            boats = self.backend.get_all_boats()
            analysis = {
                "timestamp": time.time(),
                "total_vessels": len(boats),
                "threat_summary": {
                    "neutral": 0,
                    "possible": 0,
                    "confirmed": 0
                },
                "detected_patterns": [],
                "anomalies": [],
                "overall_threat_level": "low",
                "recommendations": []
            }

            # Analyze each vessel
            for boat_id, boat in boats.items():
                threat_level = boat.get("threat_level", "neutral")
                analysis["threat_summary"][threat_level] += 1

                # Detect suspicious patterns
                patterns = self._detect_suspicious_patterns(boat, boats)
                analysis["detected_patterns"].extend(patterns)

            # Calculate overall threat level
            analysis["overall_threat_level"] = self._calculate_overall_threat(analysis["threat_summary"])

            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)

            return analysis

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "threat_summary": {"neutral": 0, "possible": 0, "confirmed": 0},
                "overall_threat_level": "unknown"
            }

    def _detect_suspicious_patterns(self, vessel: Dict, all_vessels: Dict) -> List[Dict]:
        """Detect suspicious behavior patterns in vessel"""
        patterns = []

        # Check speed anomalies
        speed = math.sqrt(vessel["velocity"][0]**2 + vessel["velocity"][1]**2)
        if speed > 6.0 and vessel.get("threat_level") == "neutral":
            patterns.append({
                "type": "suspicious_movement",
                "vessel_id": vessel["id"],
                "description": f"High speed ({speed:.1f} units/sec) for neutral vessel",
                "confidence": 0.7
            })

        # Check proximity to other vessels (potential coordination)
        if vessel.get("threat_level") in ["possible", "confirmed"]:
            close_vessels = self._find_nearby_vessels(vessel, all_vessels, radius=100)
            if len(close_vessels) >= 2:
                patterns.append({
                    "type": "group_coordination",
                    "vessel_id": vessel["id"],
                    "description": f"Moving in proximity to {len(close_vessels)} other vessels",
                    "confidence": 0.6
                })

        # Check for weapon signatures
        weapons = vessel.get("weapons", [])
        if weapons and vessel.get("threat_level") == "neutral":
            patterns.append({
                "type": "weapon_signatures",
                "vessel_id": vessel["id"],
                "description": f"Neutral vessel carrying weapons: {weapons}",
                "confidence": 0.8
            })

        return patterns

    def _find_nearby_vessels(self, target_vessel: Dict, all_vessels: Dict, radius: float) -> List[str]:
        """Find vessels within specified radius of target"""
        nearby = []
        target_pos = np.array(target_vessel["position"])

        for vessel_id, vessel in all_vessels.items():
            if vessel_id == target_vessel["id"]:
                continue

            vessel_pos = np.array(vessel["position"])
            distance = np.linalg.norm(vessel_pos - target_pos)

            if distance <= radius:
                nearby.append(vessel_id)

        return nearby

    def _calculate_overall_threat(self, threat_summary: Dict) -> str:
        """Calculate overall threat level based on vessel distribution"""
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

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate tactical recommendations based on analysis"""
        recommendations = []
        threat_level = analysis["overall_threat_level"]

        if threat_level == "high":
            recommendations.extend([
                "Immediate threat interception recommended",
                "Maintain defensive posture",
                "Consider requesting backup"
            ])
        elif threat_level == "medium":
            recommendations.extend([
                "Monitor suspicious vessels closely",
                "Maintain patrol patterns",
                "Prepare for potential escalation"
            ])
        else:
            recommendations.extend([
                "Continue routine patrol",
                "Verify identification of unknown vessels",
                "Maintain situational awareness"
            ])

        # Add pattern-specific recommendations
        for pattern in analysis["detected_patterns"]:
            if pattern["type"] == "group_coordination":
                recommendations.append("Watch for coordinated attacks")
            elif pattern["type"] == "weapon_signatures":
                recommendations.append("Approach armed vessels with caution")

        return recommendations

    def decide_action(self, vessel_id: str) -> AIReport:
        """
        Make intelligent decision about a specific vessel
        Returns detailed report with recommended action
        """
        try:
            vessel = self.backend.get_boat_details(vessel_id)
            if not vessel:
                return self._create_error_report(vessel_id, "Vessel not found")

            # Analyze vessel characteristics
            threat_assessment = self._assess_vessel_threat(vessel)
            action = self._determine_best_action(vessel, threat_assessment)
            confidence = self._calculate_confidence(vessel, action)
            reasoning = self._generate_reasoning(vessel, action, threat_assessment)

            # Create report
            report = AIReport(
                vessel_id=vessel_id,
                threat_assessment=threat_assessment,
                recommended_action=action,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=time.time()
            )

            # Store decision in history
            self.history.append({
                "vessel_id": vessel_id,
                "action": action.value,
                "confidence": confidence,
                "timestamp": report.timestamp
            })

            return report

        except Exception as e:
            return self._create_error_report(vessel_id, str(e))

    def _assess_vessel_threat(self, vessel: Dict) -> str:
        """Assess threat level of a specific vessel"""
        base_threat = vessel.get("threat_level", "neutral")
        threat_score = 0.0

        # Base threat score
        if base_threat == "confirmed":
            threat_score += 0.8
        elif base_threat == "possible":
            threat_score += 0.4

        # Weapon-based threat
        weapons = vessel.get("weapons", [])
        if weapons:
            # A neutral ship with weapons is a significant concern
            if base_threat == "neutral":
                threat_score += 0.5
            else:
                threat_score += len(weapons) * 0.1

        # Speed-based threat
        speed = math.sqrt(vessel["velocity"][0]**2 + vessel["velocity"][1]**2)
        if speed > 5.0:
            threat_score += 0.2

        # Determine threat assessment
        if threat_score >= 0.8:
            return "high"
        elif threat_score >= 0.5:
            return "medium"
        else:
            return "low"

    def _determine_best_action(self, vessel: Dict, threat_assessment: str) -> AIAction:
        """Determine the best action based on vessel assessment"""
        threat_level = vessel.get("threat_level", "neutral")

        if threat_level == "confirmed" or threat_assessment == "high":
            return AIAction.INTERCEPT
        elif self._has_suspicious_characteristics(vessel):
            return AIAction.INTERCEPT # More aggressive: armed neutral is treated as hostile
        elif threat_level == "possible" or threat_assessment == "medium":
            return AIAction.MONITOR
        else:
            return AIAction.SAFE

    def _has_suspicious_characteristics(self, vessel: Dict) -> bool:
        """Check if vessel has suspicious characteristics despite neutral threat level"""
        # Armed neutral vessel
        if vessel.get("weapons") and vessel.get("threat_level") == "neutral":
            return True

        # High speed for civilian vessel
        speed = math.sqrt(vessel["velocity"][0]**2 + vessel["velocity"][1]**2)
        vessel_type = vessel.get("vessel_type", "").lower()
        if speed > 4.0 and ("cargo" in vessel_type or "fishing" in vessel_type):
            return True

        return False

    def _calculate_confidence(self, vessel: Dict, action: AIAction) -> float:
        """Calculate confidence level for the recommended action"""
        confidence = 0.7  # Base confidence

        # Increase confidence based on clear threat indicators
        if vessel.get("threat_level") == "confirmed":
            confidence += 0.2
        if vessel.get("weapons"):
            confidence += 0.15

        # Decrease confidence for ambiguous cases
        if vessel.get("threat_level") == "neutral" and self._has_suspicious_characteristics(vessel):
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def _generate_reasoning(self, vessel: Dict, action: AIAction, threat_assessment: str) -> str:
        """Generate human-readable reasoning for the decision"""
        vessel_type = vessel.get("vessel_type", "unknown")
        threat_level = vessel.get("threat_level", "neutral")

        if action == AIAction.INTERCEPT:
            if self._has_suspicious_characteristics(vessel) and threat_level == "neutral":
                return f"Neutral vessel '{vessel_type}' exhibiting hostile indicators (weapons/speed). Recommending INTERCEPT due to high risk."
            return f"Vessel classified as {threat_level} threat with {threat_assessment} risk level. Type: {vessel_type}. Immediate interception required."
        elif action == AIAction.MONITOR:
            return f"Vessel shows potential threat indicators. Type: {vessel_type}, Threat: {threat_level}. Recommend continued monitoring."
        elif action == AIAction.SAFE:
            return f"Vessel appears to be legitimate {vessel_type} with no threat indicators. Can be marked as safe."
        else:
            return f"Insufficient data for definitive action. Recommend further observation."

    def _create_error_report(self, vessel_id: str, error: str) -> AIReport:
        """Create error report when analysis fails"""
        return AIReport(
            vessel_id=vessel_id,
            threat_assessment="unknown",
            recommended_action=AIAction.MONITOR,
            confidence=0.1,
            reasoning=f"Analysis error: {error}",
            timestamp=time.time()
        )

    def generate_status_report(self, vessel_id: str) -> Dict:
        """
        Generate comprehensive status report for UI display
        Formats AIReport as JSON-serializable dict
        """
        ai_report = self.decide_action(vessel_id)

        return {
            "vessel_id": ai_report.vessel_id,
            "threat_assessment": ai_report.threat_assessment,
            "recommended_action": ai_report.recommended_action.value,
            "confidence": ai_report.confidence,
            "reasoning": ai_report.reasoning,
            "timestamp": ai_report.timestamp,
            "report_id": f"report_{vessel_id}_{random.randint(1000, 9999)}"
        }

    def get_ai_history(self) -> List[Dict]:
        """Get history of AI decisions for analysis and learning"""
        return self.history.copy()

    def clear_history(self):
        """Clear AI decision history"""
        self.history.clear()


# Utility functions for external use
def create_ai_system(backend) -> NavalAI:
    """Factory function to create AI system with backend"""
    return NavalAI(backend)


def validate_vessel_data(vessel_data: Dict) -> bool:
    """Validate vessel data structure"""
    required_fields = ["id", "position", "velocity", "threat_level", "vessel_type"]
    return all(field in vessel_data for field in required_fields)


def calculate_interception_point(interceptor_pos: Tuple[float, float],
                               interceptor_speed: float,
                               target_pos: Tuple[float, float],
                               target_velocity: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """
    Calculate optimal interception point using quadratic formula.
    Returns the interception point (x, y) or None if interception is not possible.
    """
    p_ix, p_iy = interceptor_pos
    s_i = interceptor_speed
    p_tx, p_ty = target_pos
    v_tx, v_ty = target_velocity

    dx = p_tx - p_ix
    dy = p_ty - p_iy
    dvx = v_tx
    dvy = v_ty

    # Coefficients for the quadratic equation At^2 + Bt + C = 0
    a = dvx**2 + dvy**2 - s_i**2
    b = 2 * (dx * dvx + dy * dvy)
    c = dx**2 + dy**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No real solution, interception not possible

    # Calculate the two possible times
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Use the smallest positive time
    time_to_intercept = -1
    if t1 > 0 and t2 > 0:
        time_to_intercept = min(t1, t2)
    elif t1 > 0:
        time_to_intercept = t1
    elif t2 > 0:
        time_to_intercept = t2
    else:
        return None # No positive time solution

    interception_point = (p_tx + v_tx * time_to_intercept, p_ty + v_ty * time_to_intercept)

    return interception_point
