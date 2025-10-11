# backend.py - Enhanced with UI interaction methods and dynamic zone shrinking

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np

# -----------------------------------------------------------------------------
# Vessel dataclass
# -----------------------------------------------------------------------------

@dataclass
class Vessel:
    id: int
    vessel_type: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    threat_level: str = "unknown"
    true_threat_level: str = "neutral"
    scanned: bool = False
    active: bool = True
    distance_from_patrol: float = float("inf")
    crew_count: int = 0
    items: List[str] = field(default_factory=list)
    weapons: List[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    def update_position(self, dt: float = 1.0, bounds: Optional[Tuple[float, float]] = None):
        if not self.active:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.last_update = time.time()

        if bounds is not None:
            w, h = bounds
            self.x = max(0.0, min(w, self.x))
            self.y = max(0.0, min(h, self.y))

    def set_velocity(self, vx: float, vy: float):
        self.vx = float(vx)
        self.vy = float(vy)
        self.speed = math.hypot(self.vx, self.vy)
        if self.speed > 0:
            self.heading = math.degrees(math.atan2(self.vy, self.vx)) % 360

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vessel_type": self.vessel_type,
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "speed": float(self.speed),
            "heading": float(self.heading),
            "threat_level": self.threat_level,
            "true_threat_level": self.true_threat_level,
            "scanned": bool(self.scanned),
            "active": bool(self.active),
            "distance_from_patrol": float(self.distance_from_patrol),
            "crew_count": int(self.crew_count),
            "items": list(self.items),
            "weapons": list(self.weapons),
            "last_update": self.last_update,
        }

# -----------------------------------------------------------------------------
# FleetManager
# -----------------------------------------------------------------------------

class FleetManager:
    def __init__(self, region_size: Tuple[float, float] = (800.0, 600.0)):
        self._next_id = 1
        self.vessels: Dict[int, Vessel] = {}
        self.region_w, self.region_h = region_size

    def _generate_id(self) -> int:
        bid = self._next_id
        self._next_id += 1
        return bid

    def register_vessel(self, v: Vessel) -> Vessel:
        v.id = self._generate_id()
        self.vessels[v.id] = v
        return v

    def add_vessel(self,
                   x: float,
                   y: float,
                   vx: float = 0.0,
                   vy: float = 0.0,
                   vessel_type: str = "Cargo Ship",
                   true_threat_level: str = "neutral",
                   crew_count: int = 0,
                   items: Optional[List[str]] = None,
                   weapons: Optional[List[str]] = None) -> Vessel:
        v = Vessel(
            id=0,
            vessel_type=vessel_type,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=math.hypot(vx, vy),
            heading=(math.degrees(math.atan2(vy, vx)) % 360) if (vx != 0 or vy != 0) else 0.0,
            true_threat_level=true_threat_level,
            threat_level="unknown",
            crew_count=crew_count,
            items=items or [],
            weapons=weapons or []
        )
        return self.register_vessel(v)

    def remove_vessel(self, vid: int):
        if vid in self.vessels:
            self.vessels[vid].active = False

    def get_vessel(self, vid: int) -> Optional[Vessel]:
        return self.vessels.get(vid)

    def all_vessels(self) -> List[Vessel]:
        return list(self.vessels.values())

    def active_vessels(self) -> List[Vessel]:
        return [v for v in self.vessels.values() if v.active]

    def spawn_safe_random(self,
                          count: int,
                          region: Tuple[float, float, float, float],
                          avoid_positions: List[np.ndarray] = [],
                          min_distance: float = 30.0,
                          max_attempts: int = 300) -> List[Vessel]:
        x_min, x_max, y_min, y_max = region
        spawned: List[Vessel] = []
        attempts = 0
        existing_positions = [np.array((v.x, v.y), dtype=float) for v in self.active_vessels()]

        while len(spawned) < count and attempts < max_attempts:
            attempts += 1
            x = float(random.uniform(x_min, x_max))
            y = float(random.uniform(y_min, y_max))
            pos = np.array((x, y), dtype=float)
            bad = False

            for ap in avoid_positions:
                if np.linalg.norm(pos - ap) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            for s in spawned:
                if math.hypot(x - s.x, y - s.y) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            for ep in existing_positions:
                if np.linalg.norm(pos - ep) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            heading = random.uniform(0, 360)
            speed = random.uniform(0.3, 1.6)
            vx = math.cos(math.radians(heading)) * speed
            vy = math.sin(math.radians(heading)) * speed
            vtype = random.choice(["Fishing Boat", "Cargo Ship", "Speedboat", "Patrol Craft"])
            threat = random.choices(["neutral", "possible", "confirmed"], weights=[0.6, 0.3, 0.1])[0]
            crew = random.randint(2, 25)

            new_v = self.add_vessel(x=x, y=y, vx=vx, vy=vy, vessel_type=vtype, true_threat_level=threat, crew_count=crew)
            spawned.append(new_v)
            existing_positions.append(np.array((x, y), dtype=float))

        return spawned

# -----------------------------------------------------------------------------
# HailMessageGenerator - Handles vessel communication logic
# -----------------------------------------------------------------------------

class HailMessageGenerator:
    """Generates hail messages and responses based on vessel threat level"""

    @staticmethod
    def generate_hail_response(vessel: Vessel) -> Tuple[str, str, bool]:
        """
        Generate hail and response messages for a vessel.
        Returns: (hail_message, response_message, is_suspicious)
        """
        hail_message = "Unidentified vessel, this is Naval Patrol. Identify yourself immediately."
        is_suspicious = False
        threat = vessel.true_threat_level
        vessel_type = vessel.vessel_type

        if threat == "confirmed":
            is_suspicious = True
            responses = [
                f"This is the warship '{vessel_type}'. Stay clear or you will be fired upon!",
                "...",
                "[Radio Silence]",
                "[Static followed by weapon system charging sounds]"
            ]
            response_message = random.choice(responses)
        elif threat == "possible":
            is_suspicious = True
            responses = [
                f"This is private vessel '{vessel_type}'. State your intentions.",
                "We are on a private charter. We do not need to identify.",
                "...Stand by... We are experiencing engine trouble."
            ]
            response_message = random.choice(responses)
        else: # neutral
            is_suspicious = False
            responses = [
                f"This is the fishing vessel '{vessel_type}'. Just hauling in a catch, over.",
                f"Roger that, patrol. This is '{vessel_type}', all is well.",
                f"Hey there! This is the '{vessel_type}', just enjoying the day."
            ]
            response_message = random.choice(responses)

        return hail_message, response_message, is_suspicious

# -----------------------------------------------------------------------------
# SimulationController
# -----------------------------------------------------------------------------

class SimulationController:
    INTERCEPT_RANGE = 150

    def __init__(self, mission_type: str = "Patrol Boat", difficulty: str = "novice", player_data: Dict = None):
        self.mission_type = mission_type
        self.difficulty = difficulty
        self.player_data = player_data or {}
        self.fleet = FleetManager(region_size=(800.0, 600.0))
        self.status_log: List[str] = []
        self.selected_unit: Optional[Vessel] = None
        self.game_over: bool = False
        self.paused: bool = True
        self.patrol_phase_active: bool = True
        self.in_patrol_zone: bool = False

        # Enhanced zone management with shrinking capability
        self.zone_rect = {"x": 300, "y": 200, "width": 200, "height": 200}
        self.original_zone_rect = {"x": 300, "y": 200, "width": 200, "height": 200}  # Store original
        self.zone_expanded = False  # Track if zone was expanded

        # Create player ship BEFORE registering to set ID manually
        self.player_ship = Vessel(
            id=-1,  # Temporary ID, will be replaced
            vessel_type="Player Vessel",
            x=100.0,
            y=100.0,
            vx=0.0,
            vy=0.0,
            speed=2.0,
            crew_count=5
        )
        # Register will assign proper ID
        self.fleet.register_vessel(self.player_ship)
        print(f"[BACKEND DEBUG] Player ship created with ID: {self.player_ship.id}")

        self.units: List[Vessel] = [self.player_ship]
        self.temp_threat_markers: List[Tuple[float, float]] = []
        self._generated_vessels = False
        self.key_states: Dict[str, bool] = {"w": False, "a": False, "s": False, "d": False}

        self._listeners: Dict[str, List[Callable[..., None]]] = {
            "zone_expanded": [], "zone_shrunk": [], "boats_spawned": [], "tick": []
        }

        self.ai_generate_fn: Optional[Callable[[int, Tuple[float, float, float, float]], List[Dict[str, Any]]]] = None
        self.enemy_separation = 25.0
        self.enemy_max_speed = 1.8
        self.zone_expand_padding = (200.0, 150.0)
        self.threat_upgrade_prob_per_tick = 0.002

    # ------------------------
    # Event API
    # ------------------------

    def on(self, event_name: str, cb: Callable[..., None]):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(cb)

    def _emit(self, event_name: str, *args, **kwargs):
        for cb in self._listeners.get(event_name, []):
            try:
                cb(*args, **kwargs)
            except Exception:
                pass

    # ------------------------
    # Logging
    # ------------------------

    def add_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.status_log.append(entry)
        if len(self.status_log) > 500:
            self.status_log.pop(0)

    def get_status_log(self) -> List[str]:
        """Get the status log entries"""
        return self.status_log.copy()

    # ------------------------
    # Movement API
    # ------------------------

    def set_key_state(self, key: str, pressed: bool):
        if key in ("up", "down", "left", "right"):
            km = {"up": "w", "down": "s", "left": "a", "right": "d"}
            key = km[key]
        if key in self.key_states:
            self.key_states[key] = bool(pressed)

    def stop_player_movement(self):
        """Stop all player movement (space key)"""
        for k in self.key_states.keys():
            self.key_states[k] = False
        self.player_ship.vx = 0.0
        self.player_ship.vy = 0.0

    def _apply_key_velocity(self):
        dx = 0.0
        dy = 0.0
        if self.key_states.get("w"):
            dy -= 1.0
        if self.key_states.get("s"):
            dy += 1.0
        if self.key_states.get("a"):
            dx -= 1.0
        if self.key_states.get("d"):
            dx += 1.0

        if dx == 0 and dy == 0:
            self.player_ship.vx = 0.0
            self.player_ship.vy = 0.0
        else:
            norm = math.hypot(dx, dy)
            if norm == 0:
                vx, vy = 0.0, 0.0
            else:
                speed = getattr(self.player_ship, "speed", 2.0)
                vx = (dx / norm) * speed
                vy = (dy / norm) * speed
            self.player_ship.set_velocity(vx, vy)

    # ------------------------
    # ENHANCED Zone detection & management with shrinking
    # ------------------------

    def _patrol_in_zone(self) -> bool:
        zr = self.zone_rect
        px, py = self.player_ship.x, self.player_ship.y
        return (zr["x"] <= px <= zr["x"] + zr["width"]) and (zr["y"] <= py <= zr["y"] + zr["height"])

    def _expand_zone(self):
        """Expand the patrol zone when player enters"""
        x_min = self.zone_rect["x"]
        x_max = self.zone_rect["x"] + self.zone_rect["width"]
        y_min = self.zone_rect["y"]
        y_max = self.zone_rect["y"] + self.zone_rect["height"]

        pad_x, pad_y = self.zone_expand_padding
        new_region = (
            max(0, x_min - pad_x),
            min(self.fleet.region_w, x_max + pad_x),
            max(0, y_min - pad_y),
            min(self.fleet.region_h, y_max + pad_y),
        )

        self.zone_rect = {
            "x": new_region[0],
            "y": new_region[2],
            "width": new_region[1] - new_region[0],
            "height": new_region[3] - new_region[2],
        }

        self.zone_expanded = True
        self.temp_threat_markers.clear()
        self._emit("zone_expanded", self.zone_rect)
        self.add_log("Patrol zone expanded; revealing regional contacts.")

    def _shrink_zone(self):
        """Shrink the patrol zone back to original when player leaves"""
        if self.zone_expanded:
            self.zone_rect = self.original_zone_rect.copy()
            self.zone_expanded = False

            # Remove vessels that are now outside the original zone
            vessels_to_remove = []
            for v in self.units:
                if v is self.player_ship:
                    continue
                if not self._vessel_in_original_zone(v):
                    vessels_to_remove.append(v)

            for v in vessels_to_remove:
                v.active = False
                self.units.remove(v)

            self._emit("zone_shrunk", self.zone_rect)
            self.add_log(f"Patrol zone contracted. {len(vessels_to_remove)} contacts lost outside operational area.")

    def _vessel_in_original_zone(self, vessel: Vessel) -> bool:
        """Check if vessel is within the original zone boundaries"""
        oz = self.original_zone_rect
        return (oz["x"] <= vessel.x <= oz["x"] + oz["width"]) and (oz["y"] <= vessel.y <= oz["y"] + oz["height"])

    # ------------------------
    # Vessel generation
    # ------------------------

    def generate_random_vessels(self, count: int = 6, min_distance: float = 40.0) -> List[Vessel]:
        """Generate random vessels in the patrol zone"""
        x_min = self.zone_rect["x"]
        x_max = self.zone_rect["x"] + self.zone_rect["width"]
        y_min = self.zone_rect["y"]
        y_max = self.zone_rect["y"] + self.zone_rect["height"]

        region = (x_min, x_max, y_min, y_max)
        avoid = [np.array((self.player_ship.x, self.player_ship.y), dtype=float)]

        print(f"[DEBUG] Generating {count} vessels in region: {region}")
        print(f"[DEBUG] Player position: ({self.player_ship.x}, {self.player_ship.y})")

        spawned = self.fleet.spawn_safe_random(count=count, region=region,
                                               avoid_positions=avoid, min_distance=min_distance)

        print(f"[DEBUG] Successfully spawned {len(spawned)} vessels")
        for v in spawned:
            self.units.append(v)
            print(f"[DEBUG] Added vessel {v.id}: {v.vessel_type} at ({v.x:.1f}, {v.y:.1f})")

        if self.ai_generate_fn:
            try:
                details = self.ai_generate_fn(len(spawned), region)
                for v, det in zip(spawned, details):
                    if not det:
                        continue
                    if "crew_count" in det:
                        v.crew_count = int(det["crew_count"])
                    if "items" in det:
                        v.items = list(det["items"])
                    if "weapons" in det:
                        v.weapons = list(det["weapons"])
            except:
                pass

        self._emit("boats_spawned", [v.to_dict() for v in spawned])
        self._generated_vessels = True
        self.add_log(f"Generated {len(spawned)} vessels in patrol zone.")
        print(f"[DEBUG] Total active units: {len([v for v in self.units if v.active])}")
        return spawned

    # ------------------------
    # Enemy movement
    # ------------------------

    def _update_enemy_movement(self, dt: float = 1.0):
        active_enemies = [v for v in self.units if v.active and v is not self.player_ship]
        n = len(active_enemies)
        if n == 0:
            return

        for i, v in enumerate(active_enemies):
            repel = np.array((0.0, 0.0), dtype=float)
            pos_i = np.array((v.x, v.y), dtype=float)

            for j, other in enumerate(active_enemies):
                if i == j:
                    continue
                pos_j = np.array((other.x, other.y), dtype=float)
                vec = pos_i - pos_j
                dist = np.linalg.norm(vec)
                if dist == 0:
                    repel += np.random.uniform(-0.5, 0.5, size=2)
                elif dist < self.enemy_separation:
                    repel += (vec / (dist + 1e-6)) * (self.enemy_separation - dist) * 0.06

            wander = np.random.uniform(-0.03, 0.03, size=2)
            new_vel = np.array((v.vx, v.vy), dtype=float) + (repel + wander) * 0.5
            speed = np.linalg.norm(new_vel)
            if speed > self.enemy_max_speed:
                new_vel = (new_vel / speed) * self.enemy_max_speed
            v.vx, v.vy = float(new_vel[0]), float(new_vel[1])
            v.update_position(dt=dt, bounds=(self.fleet.region_w, self.fleet.region_h))

    # ------------------------
    # Threat dynamics
    # ------------------------

    def _update_threat_states(self):
        for v in self.units:
            if not v.active or v is self.player_ship:
                continue
            if v.true_threat_level == "possible":
                if random.random() < self.threat_upgrade_prob_per_tick:
                    v.true_threat_level = "confirmed"
                    self.add_log(f"Contact {v.vessel_type} (id={v.id}) escalated to CONFIRMED.")
            elif v.true_threat_level == "neutral":
                if random.random() < 0.0005:
                    v.true_threat_level = "possible"
                    self.add_log(f"Contact {v.vessel_type} (id={v.id}) changed to POSSIBLE.")

    # ------------------------
    # ENHANCED Simulation tick with dynamic zone management
    # ------------------------

    def update_simulation(self):
        self._apply_key_velocity()
        self.player_ship.update_position(dt=1.0, bounds=(self.fleet.region_w, self.fleet.region_h))
    # Extra clamp to prevent any overshoot at high dt or speeds
        self.player_ship.x = max(0.0, min(self.fleet.region_w, self.player_ship.x))
        self.player_ship.y = max(0.0, min(self.fleet.region_h, self.player_ship.y))

    # Always compute this before any zone logic
        currently_in = self._patrol_in_zone()



        # Fully dynamic zone handling: expand on enter/re-enter, shrink on exit
# Expand on first enter or any re-enter of the original red zone
        if currently_in and not getattr(self, "zone_expanded", False):
            self.in_patrol_zone = True
            self.patrol_phase_active = False
            self.paused = False
    # Remember original rect on first expansion only
        if not hasattr(self, "original_zone_rect"):
            self.original_zone_rect = self.zone_rect.copy()
            self._expand_zone()

    # Spawn only if very few active non-player vessels exist to avoid duplicates
        active_non_player = [v for v in self.units if v.active and v is not self.player_ship]
        if len(active_non_player) < 3:
            self.generate_random_vessels(count=6)

# Shrink when leaving the original red zone
        elif (not currently_in) and getattr(self, "zone_expanded", False):
            self.in_patrol_zone = False
            self._shrink_zone()

        


# Enter or re-enter original zone → expand operational area
        if currently_in and not getattr(self, "zone_expanded", False):
            self.in_patrol_zone = True
            self.patrol_phase_active = False
            self.paused = False
    # remember original rect on first expansion only
        if not hasattr(self, "original_zone_rect"):
            self.original_zone_rect = self.zone_rect.copy()
            self._expand_zone()
    # spawn only if very few active non-player vessels exist
        active_non_player = [v for v in self.units if v.active and v is not self.player_ship]
        if len(active_non_player) < 3:
            self.generate_random_vessels(count=6)

# Left original zone while expanded → shrink back to original
        elif (not currently_in) and getattr(self, "zone_expanded", False):
            self.in_patrol_zone = False
            self._shrink_zone()

        # Handle zone shrinking when player moves out
        if not self.patrol_phase_active and self.in_patrol_zone:
            if not self._patrol_in_zone():
                self.in_patrol_zone = False
                self._shrink_zone()
            elif self._patrol_in_zone() and not self.zone_expanded:
                # Player re-entered, expand again if needed
                self.in_patrol_zone = True
                self._expand_zone()
                # Optionally regenerate vessels
                if len([v for v in self.units if v.active and v is not self.player_ship]) < 3:
                    self.generate_random_vessels(count=5)

        if self.paused or self.game_over:
            return

        if self._generated_vessels:
            self._update_enemy_movement(dt=1.0)
            self._update_threat_states()

        for v in self.units:
            v.distance_from_patrol = self.get_distance(self.player_ship, v)

        self._emit("tick", None)

    def toggle_pause(self) -> bool:
        """Toggle pause state and return current paused state"""
        self.paused = not self.paused
        if self.paused:
            self.add_log("Simulation paused.")
        else:
            self.add_log("Simulation resumed.")
        return self.paused

    def unpause(self):
        """Unpause the simulation"""
        self.paused = False
        self.add_log("Simulation started.")

    # ------------------------
    # UI Query Methods
    # ------------------------

    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.game_over

    def is_patrol_phase_active(self) -> bool:
        """Check if patrol phase is active"""
        return self.patrol_phase_active

    def is_in_patrol_zone(self) -> bool:
        """Check if player is in patrol zone"""
        return self.in_patrol_zone

    def get_zone_info(self) -> Dict[str, float]:
        """Get patrol zone information"""
        return self.zone_rect.copy()

    def get_vessel_positions(self) -> List[Dict[str, Any]]:
        """Get all vessel positions for UI rendering"""
        positions = []
        for v in self.units:
            if not v.active:
                continue
            positions.append({
                "id": v.id,
                "x": float(v.x),
                "y": float(v.y),
                "threat_level": v.threat_level,
                "true_threat_level": v.true_threat_level,
                "scanned": v.scanned,
                "active": v.active,
                "selected": (v is self.selected_unit),
                "is_player": (v.id == self.player_ship.id)  # Check by ID
            })

        print(f"[BACKEND DEBUG] Returning {len(positions)} vessel positions")
        print(f"[BACKEND DEBUG] Player ship ID is {self.player_ship.id}")
        return positions

    def get_status_info(self) -> Dict[str, Any]:
        """Get status information for UI display"""
        confirmed = sum(1 for u in self.units if u.true_threat_level == "confirmed" and u.active)
        total_possible = sum(1 for u in self.units if u.true_threat_level in ("possible", "confirmed") and u.active)
        accuracy = 0.8

        return {
            "confirmed_threats": confirmed,
            "total_threats": total_possible,
            "accuracy": accuracy
        }

    def get_status_report(self) -> str:
        """Generate status report text for UI"""
        status = self.get_status_info()
        active_count = sum(1 for u in self.units if u.active and u is not self.player_ship)

        report = (
            f"== TACTICAL OVERVIEW ==\n"
            f"Mission Type: {self.mission_type}\n"
            f"Difficulty: {self.difficulty.upper()}\n"
            f"Overall Threat Level: {'ELEVATED' if status['confirmed_threats'] > 0 else 'STANDBY'}\n"
            f"Player Accuracy: {status['accuracy']:.1%}\n"
            f"AI Status: ADAPTIVE MODE ACTIVE\n\n"
            f"== PLAYER VESSEL STATUS ==\n"
            f"Position: ({self.player_ship.x:.1f}, {self.player_ship.y:.1f})\n"
            f"Speed: {self.player_ship.speed:.2f} knots\n"
            f"Crew Size: {self.player_ship.crew_count}\n"
            f"Hull Integrity: 100%\n"
            f"Weapon Systems: ONLINE\n\n"
            f"== ENVIRONMENT ==\n"
            f"Active Contacts: {active_count}\n"
            f"Confirmed Threats: {status['confirmed_threats']}\n"
            f"Possible Threats: {status['total_threats'] - status['confirmed_threats']}\n"
            f"Patrol Zone: {'ENTERED' if self.in_patrol_zone else 'APPROACHING'}\n"
        )

        return report

    def get_nearby_ships(self) -> List[Dict[str, Any]]:
        """Get list of nearby ships for communications panel"""
        result = []
        for v in self.units:
            if v is self.player_ship:
                continue
            if not v.active:
                continue
            dist = self.get_distance(self.player_ship, v)
            result.append({
                "id": v.id,
                "vessel_type": v.vessel_type,
                "distance": dist,
                "threat_level": v.threat_level if v.scanned else "unknown",
                "speed": v.speed,
                "heading": v.heading
            })
        return result

    # ------------------------
    # Vessel Interaction
    # ------------------------

    def handle_vessel_click(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """
        Handle vessel click event.
        Returns vessel information including hail messages, or None if no vessel clicked.
        """
        for v in self.units:
            if v is self.player_ship:
                continue
            if not v.active:
                continue

            if math.hypot(x - v.x, y - v.y) <= 12.0:
                v.scanned = True
                self.selected_unit = v

                # Generate hail messages
                hail_msg, response_msg, is_suspicious = HailMessageGenerator.generate_hail_response(v)

                # Log the interaction
                self.add_log(f"Hailed {v.vessel_type}. Response: '{response_msg}'")

                # Calculate distance
                distance = self.get_distance(self.player_ship, v)

                return {
                    "vessel_type": v.vessel_type,
                    "threat_level": v.threat_level,
                    "true_threat_level": v.true_threat_level,
                    "scanned": v.scanned,
                    "distance": distance,
                    "crew_count": v.crew_count,
                    "hail_message": hail_msg,
                    "response_message": response_msg,
                    "is_suspicious": is_suspicious
                }

        self.selected_unit = None
        return None

    def intercept_vessel(self) -> Tuple[bool, Optional[str], str]:
        """Intercept the currently selected vessel"""
        if not self.selected_unit:
            return False, None, "No vessel selected."

        target = self.selected_unit
        if target.crew_count > self.player_ship.crew_count:
            return False, target.true_threat_level, "Target too large to intercept alone. Call for backup."

        was_correct = (target.true_threat_level == "confirmed")
        message = f"Intercept action taken on {target.vessel_type} (id={target.id}). Actual: {target.true_threat_level}."
        target.active = False
        self.add_log(message)
        self.selected_unit = None
        return was_correct, target.true_threat_level, message

    def mark_safe(self) -> Tuple[bool, Optional[str], str]:
        """Mark the currently selected vessel as safe"""
        if not self.selected_unit:
            return False, None, "No vessel selected."

        t = self.selected_unit
        was_correct = (t.true_threat_level == "neutral")
        t.threat_level = "neutral"
        self.add_log(f"Marked vessel id={t.id} as SAFE (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as SAFE."

    def mark_threat(self) -> Tuple[bool, Optional[str], str]:
        """Mark the currently selected vessel as threat"""
        if not self.selected_unit:
            return False, None, "No vessel selected."

        t = self.selected_unit
        was_correct = (t.true_threat_level in ("possible", "confirmed"))
        t.threat_level = "confirmed"
        self.add_log(f"Marked vessel id={t.id} as THREAT (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as THREAT."

    def generate_distress_report(self) -> str:
        """Generate a comprehensive distress report"""
        if not self.selected_unit:
            return "ERROR: No target vessel selected for distress call."

        target = self.selected_unit
        nearby_threats = [v for v in self.units if v.active and v is not self.player_ship
                          and v.true_threat_level in ("possible", "confirmed")]

        report = f"""
=== DISTRESS CALL REPORT ===
TIMESTAMP: {time.strftime("%H:%M:%S - %Y-%m-%d")}
CALLING VESSEL: {self.player_ship.vessel_type} (ID: {self.player_ship.id})
POSITION: ({self.player_ship.x:.1f}, {self.player_ship.y:.1f})

=== PRIMARY THREAT ===
VESSEL TYPE: {target.vessel_type}
VESSEL ID: {target.id}
POSITION: ({target.x:.1f}, {target.y:.1f})
DISTANCE: {self.get_distance(self.player_ship, target):.0f}m
CREW COUNT: {target.crew_count}
SPEED: {target.speed:.1f} knots
HEADING: {target.heading:.0f}°
THREAT LEVEL: {target.true_threat_level.upper()}
SCANNED: {'YES' if target.scanned else 'NO'}

=== ADDITIONAL THREATS IN AREA ===
TOTAL POSSIBLE THREATS: {len(nearby_threats)}
"""

        for i, threat in enumerate(nearby_threats[:3], 1):  # Show max 3 additional threats
            if threat.id != target.id:
                report += f"""
THREAT #{i}:
- Type: {threat.vessel_type} (ID: {threat.id})
- Position: ({threat.x:.1f}, {threat.y:.1f})
- Distance from player: {self.get_distance(self.player_ship, threat):.0f}m
- Threat Level: {threat.true_threat_level.upper()}
"""

        report += f"""
=== TACTICAL SITUATION ===
PATROL ZONE STATUS: {'ACTIVE' if self.in_patrol_zone else 'INACTIVE'}
TOTAL ACTIVE VESSELS: {len([v for v in self.units if v.active])}
CONFIRMED HOSTILES: {len([v for v in self.units if v.active and v.true_threat_level == 'confirmed'])}

=== RECOMMENDATIONS ===
IMMEDIATE BACKUP REQUIRED: {'YES' if target.crew_count > self.player_ship.crew_count else 'NO'}
EVACUATION NEEDED: {'YES' if len(nearby_threats) >= 3 else 'NO'}
ENGAGEMENT AUTHORIZATION: REQUESTED

=== END REPORT ===
"""
        return report

    def distress_call(self) -> str:
        """Enhanced distress call with comprehensive reporting"""
        if not self.selected_unit:
            return "No target for distress call."

        target = self.selected_unit
        report = self.generate_distress_report()

        # Log the distress call
        self.add_log(f"DISTRESS CALL: Backup requested for {target.vessel_type} (ID: {target.id})")

        if target.crew_count > self.player_ship.crew_count * 1.5:
            self.add_log(f"Distress call sent for vessel {target.id}. Backup is on the way.")
            target.active = False
            self.selected_unit = None
            return f"Backup called for {target.vessel_type}. Threat neutralized."
        else:
            self.add_log(f"Distress call for vessel {target.id} denied. Threat is manageable.")
            return f"Distress call denied. Engage target directly."

    # ------------------------
    # Utility Methods
    # ------------------------

    @staticmethod
    def get_distance(a: Vessel, b: Vessel) -> float:
        """Calculate distance between two vessels"""
        return math.hypot(a.x - b.x, a.y - b.y)

    def get_positions_for_ui(self) -> List[Dict[str, Any]]:
        """Get vessel positions in UI-friendly format"""
        out = []
        for v in self.units:
            out.append({
                "id": v.id,
                "x": float(v.x),
                "y": float(v.y),
                "type": v.vessel_type,
                "threat": v.threat_level if v.scanned else "unknown",
                "true_threat": v.true_threat_level,
                "active": v.active
            })
        return out

    # ------------------------
    # Serialization (optional)
    # ------------------------