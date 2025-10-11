# backend.py
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
    threat_level: str = "unknown"       # what the UI/player perceives after scan
    true_threat_level: str = "neutral"  # ground truth used by scoring/AI
    scanned: bool = False
    active: bool = True
    distance_from_patrol: float = float("inf")
    # optional fields for AI augmentation
    crew_count: int = 0
    items: List[str] = field(default_factory=list)
    weapons: List[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    def update_position(self, dt: float = 1.0, bounds: Optional[Tuple[float, float]] = None):
        """Advance vessel position by velocity * dt, clamp into bounds if provided."""
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
# FleetManager (ID allocation and registry)
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
        """Register an existing Vessel object and return it with assigned id."""
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
            # soft remove so UI/DB can still inspect if needed
            self.vessels[vid].active = False

    def get_vessel(self, vid: int) -> Optional[Vessel]:
        return self.vessels.get(vid)

    def all_vessels(self) -> List[Vessel]:
        return list(self.vessels.values())

    def active_vessels(self) -> List[Vessel]:
        return [v for v in self.vessels.values() if v.active]

    def positions_array(self) -> np.ndarray:
        arr = np.array([[v.x, v.y] for v in self.active_vessels()], dtype=float)
        return arr

    def spawn_safe_random(self,
                          count: int,
                          region: Tuple[float, float, float, float],
                          avoid_positions: List[np.ndarray] = [],
                          min_distance: float = 30.0,
                          max_attempts: int = 300) -> List[Vessel]:
        """Spawn boats safely in region (x_min,x_max,y_min,y_max). Avoid given positions
        and existing vessels by min_distance. Returns list of created vessels (registered)."""
        x_min, x_max, y_min, y_max = region
        spawned: List[Vessel] = []
        attempts = 0

        # convert existing active vessel positions to numpy arrays for checks
        existing_positions = [np.array((v.x, v.y), dtype=float) for v in self.active_vessels()]

        while len(spawned) < count and attempts < max_attempts:
            attempts += 1
            x = float(random.uniform(x_min, x_max))
            y = float(random.uniform(y_min, y_max))
            pos = np.array((x, y), dtype=float)

            # too close to avoid_positions?
            bad = False
            for ap in avoid_positions:
                if np.linalg.norm(pos - ap) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            # too close to existing spawned?
            for s in spawned:
                if math.hypot(x - s.x, y - s.y) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            # too close to existing active vessels
            for ep in existing_positions:
                if np.linalg.norm(pos - ep) < min_distance:
                    bad = True
                    break
            if bad:
                continue

            # create the vessel
            heading = random.uniform(0, 360)
            speed = random.uniform(0.3, 1.6)
            vx = math.cos(math.radians(heading)) * speed
            vy = math.sin(math.radians(heading)) * speed
            vtype = random.choice(["Fishing Boat", "Cargo Ship", "Speedboat", "Patrol Craft"])
            threat = random.choices(["neutral", "possible", "confirmed"], weights=[0.6, 0.3, 0.1])[0]
            new_v = self.add_vessel(x=x, y=y, vx=vx, vy=vy, vessel_type=vtype, true_threat_level=threat)
            spawned.append(new_v)
            existing_positions.append(np.array((x, y), dtype=float))
        return spawned

    # optional serialization helpers (not required by UI right now)
    def export_to_json(self, filename: str):
        import json
        data = [v.to_dict() for v in self.all_vessels()]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filename: str):
        import json
        with open(filename, "r") as f:
            data = json.load(f)
        for item in data:
            v = Vessel(
                id=0,
                vessel_type=item.get("vessel_type", "Unknown"),
                x=item.get("x", 0.0),
                y=item.get("y", 0.0),
                vx=item.get("vx", 0.0),
                vy=item.get("vy", 0.0),
                speed=item.get("speed", 0.0),
                heading=item.get("heading", 0.0),
                threat_level=item.get("threat_level", "unknown"),
                true_threat_level=item.get("true_threat_level", "neutral"),
                scanned=item.get("scanned", False),
                active=item.get("active", True),
                distance_from_patrol=item.get("distance_from_patrol", float("inf")),
            )
            self.register_vessel(v)


# -----------------------------------------------------------------------------
# SimulationController (API used by ui.py)
# -----------------------------------------------------------------------------
class SimulationController:
    INTERCEPT_RANGE = 150  # pixels/meters (UI interprets as units)

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

        # zone rect used by ui.py
        self.zone_rect = {"x": 300, "y": 200, "width": 200, "height": 200}

        # player vessel: create and register via fleet
        self.player_ship = Vessel(id=0, vessel_type="Player Vessel", x=100.0, y=100.0, vx=0.0, vy=0.0, speed=2.0)
        self.fleet.register_vessel(self.player_ship)

        # units is an ordered list used throughout UI (player first)
        self.units: List[Vessel] = [self.player_ship]

        # temp markers (red dots) visible in patrol phase; cleared on expansion
        self.temp_threat_markers: List[Tuple[float, float]] = []

        # internal flags
        self._generated_vessels = False

        # key state map for continuous movement (supports hold behavior)
        # keys: 'w','a','s','d' or 'up','down','left','right'
        self.key_states: Dict[str, bool] = {"w": False, "a": False, "s": False, "d": False}

        # event listeners: zone_expanded, boats_spawned, tick
        self._listeners: Dict[str, List[Callable[..., None]]] = {"zone_expanded": [], "boats_spawned": [], "tick": []}

        # optional AI augmentation callback signature: (count, region) -> list[dict]
        self.ai_generate_fn: Optional[Callable[[int, Tuple[float, float, float, float]], List[Dict[str, Any]]]] = None

        # tuneable parameters
        self.enemy_separation = 25.0
        self.enemy_max_speed = 1.8
        self.zone_expand_padding = (200.0, 150.0)
        self.threat_upgrade_prob_per_tick = 0.002  # suspect -> confirmed probability per tick

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
                # keep simulation robust
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

    # ------------------------
    # Backwards-compatible movement API and new key API
    # ------------------------
    def move_player(self, direction: str):
        """Backwards-compatible call used by your current ui.py (calls on keypress).
        For continuous movement, this sets the internal key-state to True.
        direction: 'w','a','s','d' or 'space' (space stops movement)
        """
        if direction == "space":
            # stop all movement
            for k in self.key_states.keys():
                self.key_states[k] = False
            self.player_ship.vx = 0.0
            self.player_ship.vy = 0.0
        else:
            # set key pressed, will be processed on tick
            if direction in ("w", "a", "s", "d"):
                self.key_states[direction] = True
            else:
                # allow arrow semantics
                kmap = {"up": "w", "down": "s", "left": "a", "right": "d"}
                if direction in kmap:
                    self.key_states[kmap[direction]] = True

    def set_key_state(self, key: str, pressed: bool):
        """Recommended: call from ui on keyPress/keyRelease.
        key: 'w','a','s','d' or 'up','down','left','right'
        """
        if key in ("up", "down", "left", "right"):
            km = {"up": "w", "down": "s", "left": "a", "right": "d"}
            key = km[key]
        if key in self.key_states:
            self.key_states[key] = bool(pressed)

    def _apply_key_velocity(self):
        """Compute a velocity vector for the player based on current key_states.
        Called each tick so holding keys produces continuous motion."""
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
            # stop
            self.player_ship.vx = 0.0
            self.player_ship.vy = 0.0
        else:
            # normalize and scale by player speed
            norm = math.hypot(dx, dy)
            if norm == 0:
                vx, vy = 0.0, 0.0
            else:
                speed = getattr(self.player_ship, "speed", 2.0)
                vx = (dx / norm) * speed
                vy = (dy / norm) * speed
            self.player_ship.set_velocity(vx, vy)

    # ------------------------
    # Zone detection & expansion
    # ------------------------
    def _patrol_in_zone(self) -> bool:
        zr = self.zone_rect
        px, py = self.player_ship.x, self.player_ship.y
        return (zr["x"] <= px <= zr["x"] + zr["width"]) and (zr["y"] <= py <= zr["y"] + zr["height"])

    def _expand_zone(self):
        """Expand zone (called once). Emits zone_expanded event and clears temporary markers."""
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
        # update zone_rect to expanded rectangle for later spawn calculations / UI
        self.zone_rect = {
            "x": new_region[0],
            "y": new_region[2],
            "width": new_region[1] - new_region[0],
            "height": new_region[3] - new_region[2],
        }
        # clear UI red markers (these are temp UI markers)
        self.temp_threat_markers.clear()
        self._emit("zone_expanded", self.zone_rect)
        self.add_log("Patrol zone expanded; revealing regional contacts.")

    # ------------------------
    # Vessel generation (AI-augmented)
    # ------------------------
    def generate_random_vessels(self, count: int = 6, min_distance: float = 40.0) -> List[Vessel]:
        """Create vessels inside current (possibly expanded) zone with collision-safe placement."""
        x_min = self.zone_rect["x"]
        x_max = self.zone_rect["x"] + self.zone_rect["width"]
        y_min = self.zone_rect["y"]
        y_max = self.zone_rect["y"] + self.zone_rect["height"]
        region = (x_min, x_max, y_min, y_max)
        avoid = [np.array((self.player_ship.x, self.player_ship.y), dtype=float)]
        spawned = self.fleet.spawn_safe_random(count=count, region=region, avoid_positions=avoid, min_distance=min_distance)
        # add spawned to units list for UI consumption
        for v in spawned:
            self.units.append(v)
        # AI augmentation callback: allow external AI to fill in crew/items/weapons/threat/velocity
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
                        # set both true threat and perceived threat to reflect AI label
                        v.true_threat_level = det["threat_level"]
                    if "velocity" in det:
                        vx, vy = det["velocity"]
                        v.set_velocity(float(vx), float(vy))
            except Exception:
                # fail-safe: ignore AI errors
                pass

        self._generated_vessels = True
        self._emit("boats_spawned", [v.to_dict() for v in spawned])
        self.add_log(f"Spawned {len(spawned)} vessels in region.")
        return spawned

    # ------------------------
    # Enemy movement & separation avoidance
    # ------------------------
    def _update_enemy_movement(self, dt: float = 1.0):
        active_enemies = [v for v in self.units if v.active and v is not self.player_ship]
        n = len(active_enemies)
        if n == 0:
            return

        # positions array for quick checks
        for i, v in enumerate(active_enemies):
            repel = np.array((0.0, 0.0), dtype=float)
            pos_i = np.array((v.x, v.y), dtype=float)
            # neighbor repulsion
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

            # small wandering
            wander = np.random.uniform(-0.03, 0.03, size=2)
            new_vel = np.array((v.vx, v.vy), dtype=float) + (repel + wander) * 0.5

            # clamp speed
            speed = np.linalg.norm(new_vel)
            if speed > self.enemy_max_speed:
                new_vel = (new_vel / speed) * self.enemy_max_speed

            v.vx, v.vy = float(new_vel[0]), float(new_vel[1])
            # move vessel (dt in ms, but we treat dt as "units" to match UI's update loop)
            v.update_position(dt=dt, bounds=(self.fleet.region_w, self.fleet.region_h))

    # ------------------------
    # Threat dynamics
    # ------------------------
    def _update_threat_states(self):
        # simple stochastic promotion: possible -> confirmed occasionally
        for v in self.units:
            if not v.active or v is self.player_ship:
                continue
            if v.true_threat_level == "possible":
                if random.random() < self.threat_upgrade_prob_per_tick:
                    v.true_threat_level = "confirmed"
                    self.add_log(f"Contact {v.vessel_type} (id={v.id}) escalated to CONFIRMED.")
            elif v.true_threat_level == "neutral":
                # rare change to possible
                if random.random() < 0.0005:
                    v.true_threat_level = "possible"
                    self.add_log(f"Contact {v.vessel_type} (id={v.id}) changed to POSSIBLE.")

    # ------------------------
    # Simulation tick (called by UI timer)
    # ------------------------
    def update_simulation(self):
        """Main per-frame update called by UI; computes velocities from key states,
        moves player, spawns region vessels when required, updates enemies, and emits tick."""
        # 1) Always process player input and move the player ship
        self._apply_key_velocity()
        self.player_ship.update_position(dt=1.0, bounds=(self.fleet.region_w, self.fleet.region_h))

        # 2) Handle the initial patrol phase logic
        if self.patrol_phase_active:
            if (not self.in_patrol_zone) and self._patrol_in_zone():
                self.in_patrol_zone = True
                self.patrol_phase_active = False # End the patrol phase
                self.paused = False # Unpause the simulation
                self._expand_zone()
                self.generate_random_vessels(count=8)
            return # During patrol phase, we only move the player and check for zone entry

        # If we are past the patrol phase, check for pause/game over
        if self.paused or self.game_over:
            return

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
        
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.add_log("Simulation paused.")
        else:
            self.add_log("Simulation resumed.")
        return self.paused

    # ------------------------
    # Interaction API (UI uses these)
    # ------------------------
    def select_unit(self, x: float, y: float) -> Optional[Vessel]:
        """Select a vessel by coordinates (called by UI click). Returns the vessel or None.
        Uses small selection radius to match UI visuals."""
        for v in self.units:
            if v is self.player_ship:
                continue
            if not v.active:
                continue
            if math.hypot(x - v.x, y - v.y) <= 12.0:
                v.scanned = True
                self.selected_unit = v
                return v
        self.selected_unit = None
        return None

    def intercept_vessel(self) -> Tuple[bool, Optional[str], str]:
        """Intercept action: remove target and return (was_correct, true_threat_level, message)."""
        if not self.selected_unit:
            return False, None, "No vessel selected."
        target = self.selected_unit
        was_correct = (target.true_threat_level == "confirmed")
        message = f"Intercept action taken on {target.vessel_type} (id={target.id}). Actual: {target.true_threat_level}."
        # Mark vessel inactive (intercepted)
        target.active = False
        self.add_log(message)
        self.selected_unit = None
        return was_correct, target.true_threat_level, message

    def mark_safe(self) -> Tuple[bool, Optional[str], str]:
        if not self.selected_unit:
            return False, None, "No vessel selected."
        t = self.selected_unit
        was_correct = (t.true_threat_level == "neutral")
        t.threat_level = "neutral"
        self.add_log(f"Marked vessel id={t.id} as SAFE (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as SAFE."

    def mark_threat(self) -> Tuple[bool, Optional[str], str]:
        if not self.selected_unit:
            return False, None, "No vessel selected."
        t = self.selected_unit
        was_correct = (t.true_threat_level in ("possible", "confirmed"))
        t.threat_level = "confirmed"
        self.add_log(f"Marked vessel id={t.id} as THREAT (actual={t.true_threat_level}).")
        self.selected_unit = None
        return was_correct, t.true_threat_level, f"Marked {t.vessel_type} as THREAT."

    # ------------------------
    # Status & queries
    # ------------------------
    def get_status_info(self) -> Dict[str, Any]:
        confirmed = sum(1 for u in self.units if u.true_threat_level == "confirmed" and u.active)
        total_possible = sum(1 for u in self.units if u.true_threat_level in ("possible", "confirmed") and u.active)
        # simple accuracy metric: (intercepts correct / total intercepts) - placeholder
        accuracy = 0.8
        return {"confirmed_threats": confirmed, "total_threats": total_possible, "accuracy": accuracy}

    def get_nearby_ships(self) -> List[Dict[str, Any]]:
        result = []
        for v in self.units:
            if v is self.player_ship:
                continue
            dist = self.get_distance(self.player_ship, v)
            result.append({
                "vessel_type": v.vessel_type,
                "distance": dist,
                "threat_level": v.threat_level if v.scanned else "unknown",
                "speed": v.speed,
                "heading": v.heading
            })
        return result

    @staticmethod
    def get_distance(a: Vessel, b: Vessel) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # UI-friendly positions snapshot
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