"""
backend.py - Backend logic for Naval Combat Simulation
Handles ML, AI, and simulation logic with all requested features
"""

import random
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict

# =============================================================================
# AI SYSTEM - Coastal Defender AI Brain
# =============================================================================

class CoastalDefenderAI:
    def __init__(self):
        self.player_data = []
        self.scenario_history = []
        self.learning_data = pd.DataFrame()

    def analyze_player_weaknesses(self, players: List[Dict]) -> Dict:
        """Analyze player strengths and weaknesses"""
        if not players:
            return {"weaknesses": [], "avg_accuracy": 0.5, "avg_reaction": 0.5}

        accuracies = [p['accuracy'] for p in players]
        reactions = [p['reaction_time'] for p in players]

        analysis = {
            "weaknesses": [],
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "avg_reaction": sum(reactions) / len(reactions),
            "skill_level": "beginner" if (sum(accuracies) / len(accuracies)) < 0.7 else "advanced"
        }

        if analysis["avg_accuracy"] < 0.6:
            analysis["weaknesses"].append("aiming")
        if analysis["avg_reaction"] > 0.7:
            analysis["weaknesses"].append("slow_reaction")

        return analysis

    def generate_scenario(self, players: List[Dict], difficulty: str = "medium") -> Dict:
        """Main AI function - generates adaptive scenarios"""
        analysis = self.analyze_player_weaknesses(players)

        base_params = {
            "easy": {"enemy_count": (2, 4), "enemy_speed": 0.3},
            "medium": {"enemy_count": (3, 6), "enemy_speed": 0.5},
            "hard": {"enemy_count": (5, 8), "enemy_speed": 0.8}
        }[difficulty]

        adaptive_adjustments = self._calculate_adaptive_adjustments(analysis, difficulty)

        scenario = {
            "enemy_count": random.randint(
                base_params["enemy_count"][0] + adaptive_adjustments["enemy_bonus"],
                base_params["enemy_count"][1] + adaptive_adjustments["enemy_bonus"]
            ),
            "enemy_speed_multiplier": base_params["enemy_speed"] * adaptive_adjustments["speed_multiplier"],
            "special_events": self._generate_events(analysis, difficulty),
            "ai_notes": f"Targeting weaknesses: {', '.join(analysis['weaknesses'])}",
            "difficulty_score": adaptive_adjustments["difficulty_score"]
        }

        return scenario

    def _calculate_adaptive_adjustments(self, analysis: Dict, difficulty: str) -> Dict:
        """Calculate how to adjust scenario based on player skills"""
        adjustments = {
            "enemy_bonus": 0,
            "speed_multiplier": 1.0,
            "difficulty_score": 0
        }

        skill_multiplier = 1.0
        if analysis["skill_level"] == "advanced":
            skill_multiplier = 1.3
            adjustments["enemy_bonus"] = 1
        elif analysis["skill_level"] == "beginner":
            skill_multiplier = 0.7

        weaknesses = analysis["weaknesses"]
        if "aiming" in weaknesses:
            adjustments["speed_multiplier"] *= 1.2
        if "slow_reaction" in weaknesses:
            adjustments["enemy_bonus"] += 1

        base_score = {"easy": 30, "medium": 50, "hard": 70}[difficulty]
        adjustments["difficulty_score"] = min(95, base_score * skill_multiplier)

        return adjustments

    def _generate_events(self, analysis: Dict, difficulty: str) -> List[str]:
        """Generate special events that target player weaknesses"""
        all_events = [
            "radar_jamming", "enemy_flanking", "missile_malfunction",
            "multiple_waves", "stealth_enemies", "time_pressure"
        ]

        event_count = {"easy": 0, "medium": 1, "hard": 2}[difficulty]

        weighted_events = all_events.copy()
        if "slow_reaction" in analysis["weaknesses"]:
            weighted_events.extend(["multiple_waves", "time_pressure"] * 2)

        selected_events = random.sample(weighted_events, min(event_count, len(weighted_events)))
        return selected_events

    def update_with_real_time_data(self, ml_data, performance_history):
        """Update AI model with real-time player performance"""
        if not ml_data['timestamp']:
            return

        current_data = {
            'movement_variability': self.calculate_movement_variability(ml_data),
            'avg_reaction_time': np.mean(ml_data['reaction_time']) if ml_data['reaction_time'] else 1.0,
            'accuracy': self.calculate_real_time_accuracy(performance_history),
            'aggression_level': self.calculate_aggression_level(ml_data, performance_history),
            'patrol_efficiency': self.calculate_patrol_efficiency(ml_data)
        }

        self.update_player_profile(current_data)

    def calculate_movement_variability(self, ml_data):
        """Calculate how erratically the player moves"""
        if len(ml_data['player_x']) < 2:
            return 0.5

        dx = np.diff(ml_data['player_x'])
        dy = np.diff(ml_data['player_y'])
        movements = np.sqrt(dx**2 + dy**2)
        return np.std(movements) / (np.mean(movements) + 1e-6)

    def calculate_real_time_accuracy(self, performance_history):
        """Calculate accuracy from performance history"""
        if not performance_history:
            return 0.5

        correct_decisions = sum(1 for decision in performance_history
                               if decision.get('success', False))
        total_decisions = len(performance_history)
        return correct_decisions / total_decisions if total_decisions > 0 else 0.5

    def calculate_aggression_level(self, ml_data, performance_history):
        """Calculate player aggression based on movement and decisions"""
        if not performance_history:
            return 0.5

        intercepts = sum(1 for decision in performance_history if decision.get('action') == 'intercept')
        total_actions = len(performance_history)
        aggression = intercepts / total_actions if total_actions > 0 else 0.5

        if 'player_movement_pattern' in ml_data and ml_data['player_movement_pattern']:
            chasing_ratio = ml_data['player_movement_pattern'].count('chasing') / len(ml_data['player_movement_pattern'])
            aggression = 0.7 * aggression + 0.3 * chasing_ratio

        return aggression

    def calculate_patrol_efficiency(self, ml_data):
        """Calculate how efficiently player patrols the zone"""
        if not ml_data['player_x'] or not ml_data['player_y']:
            return 0.5

        zone_center_x, zone_center_y = 400, 300
        zone_radius = 150
        in_zone_positions = 0

        for i in range(len(ml_data['player_x'])):
            distance = math.sqrt((ml_data['player_x'][i] - zone_center_x)**2 +
                               (ml_data['player_y'][i] - zone_center_y)**2)
            if distance <= zone_radius:
                in_zone_positions += 1

        efficiency = in_zone_positions / len(ml_data['player_x'])
        return efficiency

    def update_player_profile(self, current_data):
        """Dynamically update player profile based on real performance"""
        if not self.player_data:
            return

        if 'accuracy' in current_data:
            self.player_data[0]['accuracy'] = 0.8 * self.player_data[0]['accuracy'] + 0.2 * current_data['accuracy']

        if 'avg_reaction_time' in current_data:
            reaction_score = max(0, 1 - current_data['avg_reaction_time'] / 5.0)
            self.player_data[0]['reaction_time'] = 0.8 * self.player_data[0]['reaction_time'] + 0.2 * (1 - reaction_score)

# =============================================================================
# SIMULATION ENTITIES
# =============================================================================

class Unit:
    """Base class for simulation units"""
    def __init__(self, x, y, threat_level="neutral"):
        self.x = x
        self.y = y
        self.threat_level = threat_level
        self.true_threat_level = threat_level
        self.visible_threat = False
        self.active = True
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.speed = math.sqrt(self.vx**2 + self.vy**2)
        self.heading = math.degrees(math.atan2(self.vy, self.vx))
        self.vessel_type = random.choice(["Cargo Ship", "Fishing Vessel", "Patrol Boat", "Speedboat"])
        self.distance_from_base = 0
        self.scanned = False

    def update(self):
        """Update unit position"""
        if self.active:
            self.x += self.vx
            self.y += self.vy

            if self.x < 0 or self.x > 800:
                self.vx *= -1
            if self.y < 0 or self.y > 600:
                self.vy *= -1

# =============================================================================
# SIMULATION CONTROLLER
# =============================================================================

class SimulationController:
    """Backend controller for simulation logic"""
    def __init__(self, mission_type="Patrol Boat", player_profile="novice", player_data=None):
        self.mission_type = mission_type
        self.player_profile = player_profile
        self.player_data = player_data or {"accuracy": 0.5, "reaction_time": 0.5}

        self.ai_system = CoastalDefenderAI()
        self.ai_system.player_data = [self.player_data]
        self.current_scenario = None

        self.paused = True
        self.game_over = False
        self.units = []
        self.selected_unit = None
        self.player_ship = None
        self.in_patrol_zone = False
        self.zone_rect = {"x": 300, "y": 200, "width": 200, "height": 200}

        self.INTERCEPT_RANGE = 100
        self.SCAN_RANGE = 150

        self.status_log = []

        self.ml_data = {
            'timestamp': [],
            'player_x': [], 'player_y': [],
            'threats_detected': [], 'threats_neutralized': [],
            'reaction_time': [], 'accuracy': [],
            'mission_success': [], 'player_movement_pattern': [],
            'decision_making_speed': []
        }

        self.data_collection_start_time = None
        self.last_threat_detection_time = None
        self.performance_history = []

        self.generate_ai_scenario()
        self.init_simulation()

    def add_log(self, message):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")
        if len(self.status_log) > 100:
            self.status_log.pop(0)

    def generate_ai_scenario(self):
        """Generate AI-driven scenario"""
        difficulty = "hard" if self.mission_type == "Attack Vessel" else "medium"
        self.current_scenario = self.ai_system.generate_scenario([self.player_data], difficulty)
        self.add_log(f"Scenario generated: {self.current_scenario['enemy_count']} vessels detected")

    def init_simulation(self):
        """Initialize simulation units"""
        self.player_ship = Unit(100, 100, "friendly")
        self.player_ship.vx = 0
        self.player_ship.vy = 0
        self.player_ship.vessel_type = "Naval Patrol Vessel"
        self.player_ship.visible_threat = True
        self.units.append(self.player_ship)

        num_vessels = self.current_scenario['enemy_count']
        speed_multiplier = self.current_scenario['enemy_speed_multiplier']

        threat_distribution = ["neutral", "neutral", "possible", "possible", "confirmed"]
        if "slow_reaction" in self.current_scenario['ai_notes']:
            threat_distribution = ["neutral", "possible", "possible", "confirmed", "confirmed"]

        for i in range(num_vessels):
            x = random.randint(100, 700)
            y = random.randint(50, 550)
            threat = random.choice(threat_distribution)
            vessel = Unit(x, y, threat)
            vessel.true_threat_level = threat
            vessel.threat_level = "unknown"
            vessel.vx *= speed_multiplier
            vessel.vy *= speed_multiplier
            vessel.speed = math.sqrt(vessel.vx**2 + vessel.vy**2)
            self.units.append(vessel)

        self.add_log(f"Simulation initialized with {num_vessels} vessels")

    def is_in_zone(self, x, y):
        """Check if coordinates are in patrol zone"""
        zr = self.zone_rect
        return (zr["x"] <= x <= zr["x"] + zr["width"] and 
                zr["y"] <= y <= zr["y"] + zr["height"])

    def get_distance(self, unit1, unit2):
        """Calculate distance between two units"""
        return math.sqrt((unit1.x - unit2.x)**2 + (unit1.y - unit2.y)**2)

    def is_in_range(self, unit, range_type="intercept"):
        """Check if unit is in range"""
        if not self.player_ship or not unit:
            return False

        distance = self.get_distance(self.player_ship, unit)

        if range_type == "intercept":
            return distance <= self.INTERCEPT_RANGE
        elif range_type == "scan":
            return distance <= self.SCAN_RANGE

        return False

    def get_nearby_ships(self):
        """Get list of ships within scan range"""
        nearby = []
        for unit in self.units[1:]:
            if not unit.active:
                continue

            distance = self.get_distance(self.player_ship, unit)
            if distance <= self.SCAN_RANGE:
                nearby.append({
                    'unit': unit,
                    'distance': distance,
                    'vessel_type': unit.vessel_type,
                    'threat_level': unit.threat_level if unit.scanned else "Unknown",
                    'speed': unit.speed,
                    'heading': unit.heading
                })

        nearby.sort(key=lambda x: x['distance'])
        return nearby

    def scan_vessel(self, unit):
        """Scan a vessel to reveal its threat level"""
        if not unit or not self.is_in_range(unit, "scan"):
            return False

        if not unit.scanned:
            unit.scanned = True
            unit.threat_level = unit.true_threat_level
            unit.visible_threat = True
            self.add_log(f"Scanned {unit.vessel_type}: Threat level {unit.threat_level}")
            return True
        return False

    def update_simulation(self):
        """Main simulation update loop"""
        if self.paused or self.game_over:
            return False

        if self.data_collection_start_time is None:
            self.data_collection_start_time = datetime.now()
            self.add_log("Mission started")

        self.record_player_movement()

        self.player_ship.x += self.player_ship.vx
        self.player_ship.y += self.player_ship.vy
        self.player_ship.x = max(10, min(790, self.player_ship.x))
        self.player_ship.y = max(10, min(590, self.player_ship.y))

        for unit in self.units[1:]:
            if not unit.active or unit.scanned:
                continue
            if self.is_in_range(unit, "scan") and self.in_patrol_zone:
                self.scan_vessel(unit)

        for unit in self.units[1:]:
            unit.update()

        green_vessels = [u for u in self.units[1:] if u.active and u.true_threat_level == "neutral"]
        red_vessels = [u for u in self.units[1:] if u.active and u.true_threat_level == "confirmed"]

        for red in red_vessels:
            for green in green_vessels:
                if self.check_collision(red, green):
                    self.record_decision("failed_protection", "civilian", False)
                    self.add_log("CRITICAL: Hostile vessel attacked civilian ship!")
                    self.end_game(False, "A hostile vessel attacked a civilian ship!")
                    return True

        if len(self.ml_data['timestamp']) % 50 == 0:
            self.ai_system.update_with_real_time_data(self.ml_data, self.performance_history)

        prev_zone = self.in_patrol_zone
        self.in_patrol_zone = self.is_in_zone(self.player_ship.x, self.player_ship.y)

        if self.in_patrol_zone and not prev_zone:
            self.add_log("Entered patrol zone")
        elif not self.in_patrol_zone and prev_zone:
            self.add_log("Left patrol zone")

        self.check_mission_status()

        return True

    def record_player_movement(self):
        """Record player movement for ML"""
        current_time = datetime.now()
        self.ml_data['timestamp'].append(current_time)
        self.ml_data['player_x'].append(self.player_ship.x)
        self.ml_data['player_y'].append(self.player_ship.y)

        if len(self.ml_data['player_x']) > 1:
            dx = self.ml_data['player_x'][-1] - self.ml_data['player_x'][-2]
            dy = self.ml_data['player_y'][-1] - self.ml_data['player_y'][-2]
            movement_magnitude = math.sqrt(dx**2 + dy**2)

            if movement_magnitude < 1:
                pattern = "stationary"
            elif movement_magnitude < 3:
                pattern = "patrolling"
            else:
                pattern = "chasing"
        else:
            pattern = "starting"

        self.ml_data['player_movement_pattern'].append(pattern)

    def record_decision(self, action, target_threat_level, success=True):
        """Record player decisions"""
        decision_data = {
            'timestamp': datetime.now(),
            'action': action,
            'target_threat': target_threat_level,
            'success': success,
            'player_position': (self.player_ship.x, self.player_ship.y),
            'threats_remaining': sum(1 for u in self.units[1:] if u.active and u.true_threat_level == "confirmed")
        }
        self.performance_history.append(decision_data)

    def check_collision(self, unit1, unit2):
        """Check collision between units"""
        dist = math.sqrt((unit1.x - unit2.x)**2 + (unit1.y - unit2.y)**2)
        return dist < 25

    def check_mission_status(self):
        """Check if mission is complete"""
        confirmed_threats = sum(1 for u in self.units[1:] if u.active and u.true_threat_level == "confirmed")
        if confirmed_threats == 0 and not self.game_over:
            self.add_log("All threats eliminated - Mission accomplished!")
            self.end_game(True, "Mission Accomplished!\nAll threats eliminated.")

    def end_game(self, won, message):
        """End the game"""
        self.game_over = True
        self.paused = True
        self.save_ml_data()
        return won, message

    def save_ml_data(self):
        """Save ML data to CSV"""
        if not self.ml_data['timestamp']:
            return

        df = pd.DataFrame(self.ml_data)
        filename = f"naval_ml_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        self.add_log(f"ML data saved to {filename}")

        perf_df = pd.DataFrame(self.performance_history)
        perf_filename = f"naval_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        perf_df.to_csv(perf_filename, index=False)
        self.add_log(f"Performance data saved to {perf_filename}")

    def move_player(self, direction):
        """Move player ship"""
        speed = 3
        if direction == 'w':
            self.player_ship.vy = -speed
        elif direction == 's':
            self.player_ship.vy = speed
        elif direction == 'a':
            self.player_ship.vx = -speed
        elif direction == 'd':
            self.player_ship.vx = speed
        elif direction == 'space':
            self.player_ship.vx = 0
            self.player_ship.vy = 0

    def select_unit(self, x, y):
        """Select unit at coordinates"""
        if not self.in_patrol_zone:
            return None

        for unit in self.units[1:]:
            if not unit.active:
                continue

            dist = math.sqrt((unit.x - x)**2 + (unit.y - y)**2)
            if dist < 20:
                self.selected_unit = unit
                if not unit.scanned:
                    self.scan_vessel(unit)
                return unit

        self.selected_unit = None
        return None

    def intercept_vessel(self):
        """Intercept selected vessel"""
        if not self.selected_unit or not self.in_patrol_zone:
            return False, None, "Not in patrol zone"

        if not self.is_in_range(self.selected_unit, "intercept"):
            distance = self.get_distance(self.player_ship, self.selected_unit)
            return False, None, f"Out of range ({distance:.0f}m > {self.INTERCEPT_RANGE}m)"

        threat_level = self.selected_unit.true_threat_level
        is_correct = threat_level in ['possible', 'confirmed']
        self.record_decision("intercept", threat_level, is_correct)

        self.add_log(f"Intercepted {self.selected_unit.vessel_type} - Threat: {threat_level}")

        self.selected_unit.active = False
        self.selected_unit = None
        return is_correct, threat_level, "Success"

    def mark_safe(self):
        """Mark vessel as safe"""
        if not self.selected_unit or not self.in_patrol_zone:
            return False, None, "Not in patrol zone"

        threat_level = self.selected_unit.true_threat_level
        is_correct = threat_level == 'neutral'
        self.record_decision("mark_safe", threat_level, is_correct)

        self.add_log(f"Marked {self.selected_unit.vessel_type} as safe - Actual: {threat_level}")

        self.selected_unit.threat_level = "neutral"
        self.selected_unit = None
        return is_correct, threat_level, "Marked as safe"

    def mark_threat(self):
        """Mark vessel as threat"""
        if not self.selected_unit or not self.in_patrol_zone:
            return False, None, "Not in patrol zone"

        threat_level = self.selected_unit.true_threat_level
        is_correct = threat_level in ['possible', 'confirmed']
        self.record_decision("mark_threat", threat_level, is_correct)

        self.add_log(f"Marked {self.selected_unit.vessel_type} as threat - Actual: {threat_level}")

        self.selected_unit.threat_level = "confirmed"
        self.selected_unit = None
        return is_correct, threat_level, "Marked as threat"

    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        self.add_log("Simulation paused" if self.paused else "Simulation resumed")
        return self.paused

    def get_status_info(self):
        """Get current status information"""
        confirmed_threats = sum(1 for u in self.units[1:] if u.active and u.true_threat_level == "confirmed")
        total_threats = sum(1 for u in self.units[1:] if u.active and u.true_threat_level in ["confirmed", "possible"])

        accuracy = 0.5
        if self.performance_history:
            correct = sum(1 for d in self.performance_history if d.get('success', False))
            accuracy = correct / len(self.performance_history)

        return {
            "confirmed_threats": confirmed_threats,
            "total_threats": total_threats,
            "in_zone": self.in_patrol_zone,
            "accuracy": accuracy,
            "paused": self.paused,
            "game_over": self.game_over
        }
