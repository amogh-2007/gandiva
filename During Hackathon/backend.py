"""
backend.py - Enhanced with UI interaction methods and FIXED dynamic zone management

FIXED: Proper zone expansion/shrinking, boundary enforcement, and vessel spawning
"""
from ai import NavalAI
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
# HailMessageGenerator
# -----------------------------------------------------------------------------

class HailMessageGenerator:
    """Generates hail messages and responses based on vessel threat level"""

    @staticmethod
    def generate_hail_response(vessel: Vessel) -> Tuple[str, str, bool]:
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
        else:
            is_suspicious = False
            responses = [
                f"This is the fishing vessel '{vessel_type}'. Just hauling in a catch, over.",
                f"Roger that, patrol. This is '{vessel_type}', all is well.",
                f"Hey there! This is the '{vessel_type}', just enjoying the day."
            ]
            response_message = random.choice(responses)

        return hail_message, response_message, is_suspicious

# -----------------------------------------------------------------------------
# SimulationController - FIXED RADAR LOGIC
# -----------------------------------------------------------------------------

class SimulationController:
    INTERCEPT_RANGE = 150

    def get_intercept_range(self) -> float:
        """Adaptive interception range based on patrol zone size."""
        return 150 if not self.zone_expanded else 250



    def __init__(self, mission_type: str = "Patrol Boat", difficulty: str = "novice", player_data: Dict = None):
        self.mission_type = mission_type
        self.difficulty = difficulty
        self.player_data = player_data or {}
        self.ai_controller = NavalAI()

        self.fleet = FleetManager(region_size=(800.0, 600.0))
        self.status_log: List[str] = []
        self.selected_unit: Optional[Vessel] = None
        self.game_over: bool = False
        self.paused: bool = False
        # Communication log for performance analysis
        self.communication_log = []  # list of dicts: {vessel_id, player_msg, vessel_reply, threat_level, timestamp}

        # track whether player is inside the original small patrol box
        self.in_patrol_zone: bool = False

        # zone rect used by ui.py (original small zone)
        self.zone_rect = {"x": 300, "y": 200, "width": 200, "height": 200}
        # keep a copy of the original rect so we can expand/collapse correctly
        self.zone_rect_original = dict(self.zone_rect)
        # expose original zone fields for legacy helpers that reference attributes
        self.original_x = float(self.zone_rect_original["x"])
        self.original_y = float(self.zone_rect_original["y"])
        self.original_width = float(self.zone_rect_original["width"])
        self.original_height = float(self.zone_rect_original["height"])
        self.zone_rect_expanded: Optional[Dict[str, float]] = None
        self.zone_expanded: bool = False

        # UI expects this flag (true when in patrol phase / small box displayed)
        self.patrol_phase_active: bool = True

        # player vessel: create and register via fleet
        self.player_ship = Vessel(id=0, vessel_type="Player Vessel", x=100.0, y=100.0, vx=0.0, vy=0.0, speed=4.0)
        self.fleet.register_vessel(self.player_ship)

        # units is an ordered list used throughout UI (player first)
        self.units: List[Vessel] = [self.player_ship]

        # temp markers (red dots) visible in patrol phase; cleared on expansion
        self.temp_threat_markers: List[Tuple[float, float]] = []

        # internal flags
        self._generated_vessels = False
        # track ids of vessels spawned when zone expanded so we can deactivate them on collapse
        self._spawned_ids: List[int] = []

        # key state map for continuous movement (supports hold behavior)
        self.key_states: Dict[str, bool] = {"w": False, "a": False, "s": False, "d": False}

        # event listeners
        self._listeners: Dict[str, List[Callable[..., None]]] = {"zone_expanded": [], "boats_spawned": [], "tick": []}

        self.ai_generate_fn = None

        # tuning
        self.enemy_separation = 25.0
        self.enemy_max_speed = 1.8
        self.zone_expand_padding = (200.0, 150.0)
        self.threat_upgrade_prob_per_tick = 0.002

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

    def add_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.status_log.append(entry)
        if len(self.status_log) > 500:
            self.status_log.pop(0)

    def get_status_log(self) -> List[str]:
        return self.status_log.copy()

    # FIXED: Movement with boundary enforcement
    def set_key_state(self, key: str, pressed: bool):
        if key in ("up", "down", "left", "right"):
            km = {"up": "w", "down": "s", "left": "a", "right": "d"}
            key = km[key]
        if key in self.key_states:
            self.key_states[key] = bool(pressed)

    def stop_player_movement(self):
        for k in list(self.key_states.keys()):
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

    # FIXED: Zone detection - check if player touches the CURRENT zone boundary
    def _patrol_in_zone(self) -> bool:
        """Check if player is touching or inside the current zone"""
        px, py = self.player_ship.x, self.player_ship.y
        
        if not self.zone_expanded:
            # Check against original small zone
            return (self.original_x <= px <= self.original_x + self.original_width) and \
                   (self.original_y <= py <= self.original_y + self.original_height)
        else:
            # Check against expanded zone (full screen)
            return (0 <= px <= 800) and (0 <= py <= 600)

    # FIXED: Zone expansion - MODIFY the SAME zone_rect
    # --- Replace _expand_zone with this version ---
    def _expand_zone(self):
        """Expand zone (called once). Emits zone_expanded event and clears temporary markers."""
        if self.zone_expanded:
            return

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

    # store expanded rect and set as current zone_rect
        self.zone_rect_expanded = {
            "x": new_region[0],
            "y": new_region[2],
            "width": new_region[1] - new_region[0],
            "height": new_region[3] - new_region[2],
        }
        self.zone_rect = dict(self.zone_rect_expanded)
        self.zone_expanded = True

        # end patrol phase (UI uses this)
        self.patrol_phase_active = False

        # clear UI red markers (these are temp UI markers)
        self.temp_threat_markers.clear()
        self._emit("zone_expanded", self.zone_rect)
        self.add_log("Patrol zone expanded; revealing regional contacts.")

        # --- Add this new method to collapse zone dynamically ---
    def _collapse_zone(self):
        """Collapse the zone back to the original small patrol rect and clear spawned vessels."""
        if not self.zone_expanded:
            return

        # reset zone rect to original
        self.zone_rect = dict(self.zone_rect_original)
        self.zone_rect_expanded = None
        self.zone_expanded = False

        # re-enter patrol phase UI state
        self.patrol_phase_active = True

        # deactivate/soft-remove spawned vessels (so UI stops drawing them)
        for vid in list(self._spawned_ids):
            v = self.fleet.vessels.get(vid)

            if v:
                v.active = False
        self._spawned_ids.clear()

        # clean up units list so only active vessels remain (player kept)
        self.units = [self.player_ship] + [v for v in self.fleet.active_vessels() if v is not self.player_ship]

        self._generated_vessels = False
        self.temp_threat_markers.clear()
        self._emit("zone_collapsed", self.zone_rect)
        self.add_log("Patrol zone collapsed; returning to original patrol area.")


       # --- Add helper checks for original/expanded containment ---
    def _player_inside_original_zone(self) -> bool:
        zr = self.zone_rect_original
        px, py = self.player_ship.x, self.player_ship.y
        return (zr["x"] <= px <= zr["x"] + zr["width"]) and (zr["y"] <= py <= zr["y"] + zr["height"])

    def _player_inside_expanded_zone(self) -> bool:
        if not self.zone_rect_expanded:
            return False
        zr = self.zone_rect_expanded
        px, py = self.player_ship.x, self.player_ship.y
        return (zr["x"] <= px <= zr["x"] + zr["width"]) and (zr["y"] <= py <= zr["y"] + zr["height"])



    # FIXED: Zone shrinking - RESTORE the SAME zone_rect
    def _shrink_zone(self):
        """INSTANTLY shrink the SAME zone_rect back to original size"""
        if self.zone_expanded:
            # Restore original values to the same zone_rect
            self.zone_rect["x"] = self.original_x
            self.zone_rect["y"] = self.original_y
            self.zone_rect["width"] = self.original_width
            self.zone_rect["height"] = self.original_height
            
            self.zone_expanded = False

            # Remove vessels that are now outside the original zone
            vessels_to_remove = []
            for v in list(self.units):
                if v is self.player_ship:
                    continue
                if not self._vessel_in_original_zone(v):
                    vessels_to_remove.append(v)

            for v in vessels_to_remove:
                v.active = False
                if v in self.units:
                    self.units.remove(v)

            self._emit("zone_shrunk", self.zone_rect)
            self.add_log(f"Patrol zone contracted. {len(vessels_to_remove)} contacts lost outside operational area.")

    def _vessel_in_original_zone(self, vessel: Vessel) -> bool:
        """Check if vessel is within the original zone boundaries"""
        return (self.original_x <= vessel.x <= self.original_x + self.original_width) and \
               (self.original_y <= vessel.y <= self.original_y + self.original_height)

    # FIXED: Vessel generation in full screen when expanded
    # --- Replace generate_random_vessels to track spawned ids ---
    def generate_random_vessels(self, count: int = 6, min_distance: float = 40.0) -> List[Vessel]:
        """Create vessels inside current (possibly expanded) zone with collision-safe placement."""
        x_min = self.zone_rect["x"]
        x_max = self.zone_rect["x"] + self.zone_rect["width"]
        y_min = self.zone_rect["y"]
        y_max = self.zone_rect["y"] + self.zone_rect["height"]
        region = (x_min, x_max, y_min, y_max)
        avoid = [np.array((self.player_ship.x, self.player_ship.y), dtype=float)]
        spawned = self.fleet.spawn_safe_random(count=count, region=region, avoid_positions=avoid, min_distance=min_distance)

        for v in spawned:
            self.units.append(v)
            # track spawned ids for collapse cleanup
            self._spawned_ids.append(v.id)

        # AI augmentation callback remains as before (unchanged)
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
                    if "threat_level" in det:
                        v.true_threat_level = det["threat_level"]
                    if "velocity" in det:
                        vx, vy = det["velocity"]
                        v.set_velocity(float(vx), float(vy))
            except Exception:
                pass

        self._generated_vessels = True
        self._emit("boats_spawned", [v.to_dict() for v in spawned])
        self.add_log(f"Spawned {len(spawned)} vessels in region.")
        return spawned


    def _update_enemy_movement(self, dt: float = 1.0):
        """Ensure AI vessels move naturally but stay within the current patrol zone."""
        active_enemies = [v for v in self.units if v.active and v is not self.player_ship]
        if not active_enemies:
            return

        zr = self.zone_rect
        x_min, x_max = zr["x"], zr["x"] + zr["width"]
        y_min, y_max = zr["y"], zr["y"] + zr["height"]

        for i, v in enumerate(active_enemies):
            # Add gentle random motion
            wander_angle = random.uniform(0, 2 * math.pi)
            wander_speed = random.uniform(0.3, self.enemy_max_speed)
            vx, vy = math.cos(wander_angle) * wander_speed, math.sin(wander_angle) * wander_speed

            # Repel slightly from other vessels
            repel_x, repel_y = 0.0, 0.0
            for j, other in enumerate(active_enemies):
                if i == j:
                    continue
                dx, dy = v.x - other.x, v.y - other.y
                dist = math.hypot(dx, dy)
                if 0 < dist < self.enemy_separation:
                    repel_x += dx / dist * 0.25
                    repel_y += dy / dist * 0.25

            v.vx = vx + repel_x
            v.vy = vy + repel_y
            v.update_position(dt=dt)

        # Keep the vessel strictly inside the current patrol zone
            if v.x < x_min:
                v.x, v.vx = x_min, abs(v.vx)
            elif v.x > x_max:
                v.x, v.vx = x_max, -abs(v.vx)
            if v.y < y_min:
               v.y, v.vy = y_min, abs(v.vy)
            elif v.y > y_max:
                v.y, v.vy = y_max, -abs(v.vy)

            # --- AI BOUNDARY SIMULATION UPDATE ---
        zone = self.zone_rect
        x_min, x_max = zone["x"], zone["x"] + zone["width"]
        y_min, y_max = zone["y"], zone["y"] + zone["height"]
        region_bounds = (x_min, x_max, y_min, y_max)

        # Build simple vessel dicts for AI movement handling
        vessels = [
            {"x": v.x, "y": v.y, "vx": v.vx, "vy": v.vy, "speed": np.hypot(v.vx, v.vy),
            "heading": np.degrees(np.arctan2(v.vy, v.vx)), "vessel_type": v.vessel_type}
            for v in self.units if v.active and v is not self.player_ship
        ]

# Let AI simulate random motion and bounding (keeps within red region)
        self.ai_controller.update_vessel_positions(vessels, region_bounds)

# Push results back to actual vessels
        for v_dict, vessel in zip(vessels, [u for u in self.units if u.active and u is not self.player_ship]):
            vessel.x = v_dict["x"]
            vessel.y = v_dict["y"]
            vessel.vx = v_dict["vx"]
            vessel.vy = v_dict["vy"]


    def respond_to_communication(self, vessel_id: int, player_message: str) -> str:
        """
        Generate a vessel's response to a player's message based on its scenario context.
        Logs the communication for performance evaluation.
        """
        vessel = self.fleet.vessels.get(vessel_id)
        if not vessel:
            return "No response — vessel not found."

        threat = vessel.true_threat_level

        # Basic AI response logic
        if threat == "confirmed":
            # Aggressive or evasive responses
            possible_replies = [
                "Vessel refuses to comply and issues a hostile response!",
                "The vessel ignores your hails and changes course.",
                "They broadcast threats over the radio."
            ]
            reply = random.choice(possible_replies)
        elif threat == "possible":
            possible_replies = [
                "The vessel gives vague answers, avoiding your questions.",
                "They identify themselves, but their story seems inconsistent.",
                "They claim to be on a routine trip but avoid giving details."
            ]
            reply = random.choice(possible_replies)
        else:
            possible_replies = [
                "The vessel identifies itself and complies with communication.",
                "They respond politely and provide their credentials.",
                "They acknowledge your hail and offer to cooperate."
            ]
            reply = random.choice(possible_replies)

        # Log the interaction for later performance analysis
        self.communication_log.append({
            "timestamp": time.time(),
            "vessel_id": vessel_id,
            "player_msg": player_message,
            "vessel_reply": reply,
            "threat_level": threat
        })

        return reply

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

    # FIXED: Check if player is outside the original zone (to trigger shrink)
    def _player_exited_original_zone(self) -> bool:
        """Check if player has exited the original zone boundaries"""
        if not self.zone_expanded:
            return False
        
        px, py = self.player_ship.x, self.player_ship.y
        # Check if outside original zone
        return not ((self.original_x <= px <= self.original_x + self.original_width) and 
                   (self.original_y <= py <= self.original_y + self.original_height))
    
    def select_unit(self, x: float, y: float) -> Optional[Vessel]:
        """Select a vessel by coordinates (called by UI click). Returns the vessel or None."""
        for v in self.units:
            if v is self.player_ship:
                continue
            if not v.active:
                continue
        # check if the click is close enough to the vessel
            if math.hypot(x - v.x, y - v.y) <= 12.0:
                v.scanned = True
                self.selected_unit = v
                return v
        # nothing matched
        self.selected_unit = None
        return None

    # FIXED: Simulation update with proper zone management
    # --- Replace update_simulation with the version that collapses when leaving expanded zone ---
    def update_simulation(self):
        """Main per-frame update called by UI."""
        if self.paused or self.game_over:
            return

        # 1) Apply player's held keys into velocity and move player
        self._apply_key_velocity()
        self.player_ship.update_position(dt=1.0, bounds=(self.fleet.region_w, self.fleet.region_h))

        # 2) handle entering original small zone -> expand once
        if (not self.in_patrol_zone) and self._player_inside_original_zone():
            self.in_patrol_zone = True
            self._expand_zone()
            self.generate_random_vessels(count=8)

        # 2b) handle leaving expanded zone -> collapse back
        if self.zone_expanded and (not self._player_inside_expanded_zone()):
            # player left the expanded area -> collapse and return to patrol
            self._collapse_zone()
            # leaving expanded means player probably outside the original small zone too
            self.in_patrol_zone = False

        # 3) update enemy movement and separation if region is expanded / vessels exist
        if self._generated_vessels:
            self._update_enemy_movement(dt=1.0)

        # 4) threat state dynamics
        self._update_threat_states()

        # 5) recompute distances
        for v in self.units:
            v.distance_from_patrol = self.get_distance(self.player_ship, v)

        # 6) emit tick event
        self._emit("tick", None)


    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        if self.paused:
            self.add_log("Simulation paused.")
        else:
            self.add_log("Simulation resumed.")
        return self.paused

    def unpause(self):
        self.paused = False
        self.add_log("Simulation started.")

    # UI Query Methods
    def is_game_over(self) -> bool:
        return self.game_over

    def is_patrol_phase_active(self) -> bool:
        return self.patrol_phase_active

    def is_in_patrol_zone(self) -> bool:
        return self.in_patrol_zone

    def get_zone_info(self) -> Dict[str, float]:
        # Return CURRENT zone (expands/shrinks) for UI to draw
        return self.zone_rect.copy()

    def get_vessel_positions(self) -> List[Dict[str, Any]]:
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
                "is_player": (v.id == self.player_ship.id),
                "vessel_type": v.vessel_type
            })
        return positions

    def get_status_info(self) -> Dict[str, Any]:
        confirmed = sum(1 for u in self.units if u.true_threat_level == "confirmed" and u.active)
        total_possible = sum(1 for u in self.units if u.true_threat_level in ("possible", "confirmed") and u.active)
        accuracy = 0.8
        return {
            "confirmed_threats": confirmed,
            "total_threats": total_possible,
            "accuracy": accuracy
        }

    def get_status_report(self) -> str:
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

    def handle_vessel_click(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        for v in self.units:
            if v is self.player_ship:
                continue
            if not v.active:
                continue
            if math.hypot(x - v.x, y - v.y) <= 12.0:
                v.scanned = True
                self.selected_unit = v
                hail_msg, response_msg, is_suspicious = HailMessageGenerator.generate_hail_response(v)
                self.add_log(f"Hailed {v.vessel_type}. Response: '{response_msg}'")
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

    # --- Replace intercept_vessel with this corrected version ---
    def intercept_vessel(self) -> Tuple[bool, Optional[str], str]:
        """
        Intercept a selected vessel.
        Removes hostile (confirmed) vessels and logs 'Threat eliminated'.
        """
        if not self.selected_unit:
            return False, None, "No vessel selected."

        target = self.selected_unit
        # Inside range check handled in UI
        was_hostile = (target.true_threat_level == "confirmed")

        if was_hostile:
            target.active = False
            log_msg = f"⚠️ Threat eliminated: {target.vessel_type} (ID: {target.id})"
            self.add_log(log_msg)
            report_msg = "Threat eliminated."
        else:
            log_msg = f"Intercepted non-hostile vessel {target.vessel_type} (ID: {target.id})"
            self.add_log(log_msg)
            report_msg = "No threat detected – vessel secured."

        # clear selection
        self.selected_unit = None
        return was_hostile, target.true_threat_level, report_msg


    # --- Replace mark_safe / mark_threat with cleaned-up versions ---
    def mark_safe(self) -> Tuple[bool, Optional[str], str]:
        if not self.selected_unit:
            return False, None, "No vessel selected."
        t = self.selected_unit
        was_correct = (t.true_threat_level == "neutral")
        t.threat_level = "neutral"
        t.scanned = True
        self.add_log(f"Marked vessel id={t.id} as SAFE (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as SAFE."

    def mark_threat(self) -> Tuple[bool, Optional[str], str]:
        if not self.selected_unit:
            return False, None, "No vessel selected."
        t = self.selected_unit
        was_correct = (t.true_threat_level in ("possible", "confirmed"))
        t.threat_level = "confirmed"
        t.scanned = True
        self.add_log(f"Marked vessel id={t.id} as THREAT (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as THREAT."


    def generate_distress_report(self) -> str:
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
        for i, threat in enumerate(nearby_threats[:3], 1):
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
        if not self.selected_unit:
            return "No target for distress call."
        target = self.selected_unit
        report = self.generate_distress_report()
        self.add_log(f"DISTRESS CALL: Backup requested for {target.vessel_type} (ID: {target.id})")
        if target.crew_count > self.player_ship.crew_count * 1.5:
            self.add_log(f"Distress call sent for vessel {target.id}. Backup is on the way.")
            target.active = False
            self.selected_unit = None
            return f"Backup called for {target.vessel_type}. Threat neutralized."
        else:
            self.add_log(f"Distress call for vessel {target.id} denied. Threat is manageable.")
            return f"Distress call denied. Engage target directly."

    @staticmethod
    def get_distance(a: Vessel, b: Vessel) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def get_positions_for_ui(self) -> List[Dict[str, Any]]:
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