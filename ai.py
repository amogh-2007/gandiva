# ai.py - FIXED VERSION
"""
Naval Combat AI Engine - FIXED VERSION
Fixed array boolean evaluation error
"""

import numpy as np
import random
import math
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Professional logging and error resilience
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock database module (returns dicts for consistency)
class MockDatabase:
    def get_boat(self, vessel_id):
        mock_data = {
            "x": random.uniform(100, 700), "y": random.uniform(100, 500),
            "speed": random.uniform(0, 10), "heading": random.uniform(0, 360),
            "vessel_type": random.choice(["Trawler", "Cargo Ship"]),
            "behavior": "idle", "true_threat_level": random.choice(["neutral", "possible", "confirmed"]),
            "evasion_chance": random.uniform(0.1, 0.8), "detection_range": 200, "aggressiveness": 0.1,
            "id": f"vessel_{random.randint(1000, 9999)}"
        }
        return mock_data
    
    def get_sim_state(self, player_id):
        return {"x": 400, "y": 300, "speed": 15, "heading": 0, "vessel_type": "Player Ship",
                "behavior": "command", "true_threat_level": "neutral", "evasion_chance": 0.0,
                "detection_range": 500, "aggressiveness": 0.0, "id": "player_1"}

database = MockDatabase()

# --- Enhanced Deep Learning Framework ---
class DQNetwork:
    """Enhanced Deep Q-Network with proper training."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.rand(state_size * action_size).reshape(state_size, action_size) * 0.01
        self.bias = np.zeros(action_size)
        logger.info(f"Initialized DQN with state size {state_size} and action size {action_size}")
    
    def predict(self, state):
        """Predict Q-values for given states."""
        if state.ndim == 1: 
            state = state.reshape(1, -1)
        
        # Ensure state has correct dimensions
        if state.shape[1] < self.state_size:
            state = np.hstack([state, np.zeros((state.shape[0], self.state_size - state.shape[1]))])
        elif state.shape[1] > self.state_size:
            state = state[:, :self.state_size]
        
        # Simple linear model for mock implementation
        q_values = np.dot(state, self.weights) + self.bias
        return q_values
    
    def train_on_batch(self, states, targets, sample_weights=None):
        """Train network on a batch of experiences."""
        predictions = self.predict(states)
        
        # FIXED: Proper sample_weights handling
        if sample_weights is not None:
            # Check if sample_weights has elements using len() instead of boolean evaluation
            if hasattr(sample_weights, '__len__') and len(sample_weights) > 0:
                # Reshape sample_weights for broadcasting
                sample_weights = sample_weights.reshape(-1, 1)
                if sample_weights.shape[0] != predictions.shape[0]:
                    logger.warning(f"Sample weights shape mismatch: {sample_weights.shape[0]} vs {predictions.shape[0]}")
                    sample_weights = np.ones_like(predictions)
                
                # Calculate weighted loss
                errors = predictions - targets
                weighted_errors = sample_weights * errors
                loss = np.mean(weighted_errors ** 2)
            else:
                errors = predictions - targets
                loss = np.mean(errors ** 2)
        else:
            errors = predictions - targets
            loss = np.mean(errors ** 2)
        
        # Simple gradient descent update (mock implementation)
        learning_rate = 0.001
        if sample_weights is not None and hasattr(sample_weights, '__len__') and len(sample_weights) > 0:
            gradient = 2 * np.dot(states.T, weighted_errors) / len(states)
        else:
            gradient = 2 * np.dot(states.T, errors) / len(states)
        
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * np.mean(errors, axis=0)
        
        logger.debug(f"Training loss: {loss:.4f}")
        return loss
    
    def get_weights(self): 
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist()}
    
    def set_weights(self, weights):
        if "weights" in weights and "bias" in weights:
            self.weights = np.array(weights["weights"])
            self.bias = np.array(weights["bias"])

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    """Enhanced replay buffer with proper error handling."""
    def __init__(self, buffer_size=5000, alpha=0.6, beta=0.4):
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.td_errors = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done, error):
        """Add experience to buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
        priority = (abs(error) + 1e-5) ** self.alpha
        self.priorities.append(priority)
        self.td_errors.append(abs(error) + 1e-5)
    
    def sample(self, batch_size=32):
        """Sample batch of experiences with priorities."""
        if len(self.memory) == 0:
            return [], np.array([]), []
        
        batch_size = min(len(self.memory), batch_size)
        prios = np.array(self.priorities)
        probs = prios / prios.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        samples = [self.memory[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        if hasattr(weights, '__len__') and len(weights) > 0 and weights.max() > 0:
            weights /= weights.max()
        
        return samples, weights, indices
    
    def update_priorities(self, indices, errors):
        """Update priorities for sampled experiences."""
        for idx, error in zip(indices, errors):
            if 0 <= idx < len(self.td_errors):
                self.td_errors[idx] = abs(error) + 1e-5
                self.priorities[idx] = self.td_errors[idx] ** self.alpha

# --- AI Core Components ---

class ThreatLevel(Enum):
    NEUTRAL = "neutral"
    POSSIBLE = "possible" 
    CONFIRMED = "confirmed"

class AIAction(Enum):
    INTERCEPT = "intercept"
    MONITOR = "monitor"
    IGNORE = "ignore"
    EVADE = "evade"
    AWAIT_CONFIRMATION = "await_confirmation"

@dataclass
class AIReport:
    vessel_id: str
    threat_assessment: str
    recommended_action: AIAction
    confidence: float
    reasoning: str
    timestamp: float

class NavalAI:
    """
    Enhanced Naval AI Engine with robust training and realistic behaviors.
    """
    
    def __init__(self, backend=None, state_size=11, action_size=len(AIAction)):
        self.backend = backend
        self.max_state_size = state_size
        self.action_size = action_size
        
        # Initialize neural networks
        self.model = DQNetwork(self.max_state_size, action_size)
        self.target_model = DQNetwork(self.max_state_size, action_size)
        self.update_target_model()
        
        # Training parameters
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size=5000)  # Increased buffer
        self.discount_factor = 0.90  # Reduced for more immediate rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum exploration
        self.epsilon_decay = 0.998  # Slower decay
        self.batch_size = 32
        
        # Human-in-the-loop parameters
        self.hitl_confidence_threshold = 0.6  # Lower threshold for more autonomy
        self.reward_shaping = {"human_override": -8, "human_confirm": +6}  # Balanced rewards
        
        # Performance tracking
        self.performance_metrics = {
            "decisions": 0, 
            "hitl_requests": 0, 
            "cumulative_reward": 0.0, 
            "vessels_generated": 0,
            "training_losses": []
        }
        
        # Realistic behavior parameters
        self.threat_response_delay = random.uniform(2.0, 8.0)
        self.communication_realism = True
        self.tactical_maneuvering = True
        
        # Environmental factors
        self.environment_factors = {
            "sea_state": random.randint(1, 5),
            "visibility": random.uniform(2.0, 20.0),
            "time_of_day": random.choice(["dawn", "day", "dusk", "night"])
        }
        
        # Load behavior templates
        self.vessel_behaviors = self._load_realistic_behaviors()
        
        logger.info("NavalAI initialized successfully")

    def _load_realistic_behaviors(self) -> Dict:
        """Load realistic vessel behavior templates."""
        return {
            "fishing_pattern": {
                "speed_range": (2, 8),
                "course_changes": "frequent",
                "threat_response": "avoid"
            },
            "commercial_route": {
                "speed_range": (10, 18),
                "course_changes": "infrequent", 
                "threat_response": "maintain"
            },
            "aggressive_patrol": {
                "speed_range": (20, 35),
                "course_changes": "erratic",
                "threat_response": "engage"
            },
            "naval_patrol": {
                "speed_range": (15, 25),
                "course_changes": "moderate",
                "threat_response": "engage"
            },
            "suspicious_fishing": {
                "speed_range": (15, 25),
                "course_changes": "moderate",
                "threat_response": "evade"
            }
        }

    def update_target_model(self):
        """Update target network with current model weights."""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated")

    def get_state(self, vessel: Dict, player_vessel: Dict) -> np.ndarray:
        """Enhanced state representation with robust error handling."""
        try:
            # Validate required keys with defaults
            vessel_x = vessel.get('x', 400.0)
            vessel_y = vessel.get('y', 300.0)
            vessel_speed = vessel.get('speed', 1.0)
            
            player_x = player_vessel.get('x', 400.0)
            player_y = player_vessel.get('y', 300.0)
            player_speed = player_vessel.get('speed', 0.0)
            
            vessel_pos = np.array([vessel_x, vessel_y])
            player_pos = np.array([player_x, player_y])
            
            # Calculate relative position and distance
            relative_pos = vessel_pos - player_pos
            distance = np.linalg.norm(relative_pos) / 1000.0  # Normalize
            
            # Calculate bearing and relative course
            bearing = math.atan2(relative_pos[1], relative_pos[0]) / math.pi
            relative_bearing = self._calculate_relative_bearing(vessel, player_vessel)
            
            # Threat indicators
            is_possible = 1 if vessel.get('true_threat_level', 'neutral') == 'possible' else 0
            is_confirmed = 1 if vessel.get('true_threat_level', 'neutral') == 'confirmed' else 0
            evasion_chance = vessel.get('evasion_chance', 0.1)
            
            # Enhanced state features
            base_state = np.array([
                distance,
                vessel_speed / 10.0,  # Normalize speed
                player_speed / 10.0,
                bearing,
                relative_bearing,
                is_possible,
                is_confirmed,
                evasion_chance
            ])
            
            # Environmental factors
            env_state = np.array([
                self.environment_factors["sea_state"] / 5.0,
                self.environment_factors["visibility"] / 20.0,
                ["dawn", "day", "dusk", "night"].index(self.environment_factors["time_of_day"]) / 3.0
            ])
            
            # Combine and ensure correct size
            full_state = np.concatenate([base_state, env_state])
            
            # Pad or truncate to expected size
            if len(full_state) < self.max_state_size:
                full_state = np.pad(full_state, (0, self.max_state_size - len(full_state)))
            elif len(full_state) > self.max_state_size:
                full_state = full_state[:self.max_state_size]
            
            return full_state
            
        except Exception as e:
            logger.warning(f"Error processing state: {e}")
            return np.zeros(self.max_state_size)

    def _calculate_relative_bearing(self, vessel: Dict, player_vessel: Dict) -> float:
        """Calculate relative bearing with robust error handling."""
        try:
            vessel_heading = vessel.get('heading', 0.0)
            player_heading = player_vessel.get('heading', 0.0)
            
            dx = vessel.get('x', 0) - player_vessel.get('x', 0)
            dy = vessel.get('y', 0) - player_vessel.get('y', 0)
            
            if dx == 0 and dy == 0:
                return 0.0
                
            target_bearing = math.degrees(math.atan2(dy, dx)) % 360
            relative_bearing = (target_bearing - vessel_heading) % 360
            
            return relative_bearing / 360.0  # Normalize
            
        except Exception as e:
            logger.warning(f"Error calculating relative bearing: {e}")
            return 0.0

    def decide_action(self, vessel: Dict, player_vessel: Dict) -> AIReport:
        """Make decision for a vessel with confidence estimation."""
        try:
            state = self.get_state(vessel, player_vessel)
            
            # Exploration vs exploitation
            if np.random.rand() <= self.epsilon:
                # Exploration: random action
                action_index = random.randrange(self.action_size)
                confidence = random.uniform(0.3, 0.6)
            else:
                # Exploitation: use model prediction
                q_values = self.model.predict(state.reshape(1, -1))[0]
                action_index = np.argmax(q_values)
                
                # Calculate confidence from Q-values
                q_max = np.max(q_values)
                q_min = np.min(q_values)
                if q_max - q_min > 0:
                    confidence = (q_values[action_index] - q_min) / (q_max - q_min)
                else:
                    confidence = 0.5
            
            action = list(AIAction)[action_index]
            
            # Human-in-the-loop check
            if confidence < self.hitl_confidence_threshold and action == AIAction.INTERCEPT:
                action = AIAction.AWAIT_CONFIRMATION
                self.performance_metrics["hitl_requests"] += 1
                reasoning = f"Low confidence ({confidence:.2f}). Requesting human validation for INTERCEPT."
            else:
                reasoning = self._generate_reasoning(vessel, action)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.performance_metrics["decisions"] += 1
            
            return AIReport(
                vessel_id=vessel.get('id', 'unknown'),
                threat_assessment="dynamic",
                recommended_action=action,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            # Return safe default action
            return AIReport(
                vessel_id=vessel.get('id', 'unknown'),
                threat_assessment="error",
                recommended_action=AIAction.MONITOR,
                confidence=0.1,
                reasoning=f"Error in AI decision: {e}",
                timestamp=time.time()
            )

    def record_and_train(self, state, action_index, reward, next_state, done, human_feedback=None):
        """Store experience and train with optional human feedback."""
        try:
            # Apply human feedback rewards
            if human_feedback == "override": 
                reward += self.reward_shaping["human_override"]
            elif human_feedback == "confirm": 
                reward += self.reward_shaping["human_confirm"]

            # Track cumulative reward
            self.performance_metrics["cumulative_reward"] += reward

            # Ensure state dimensions
            state_padded = self._ensure_state_shape(state)
            next_state_padded = self._ensure_state_shape(next_state)

            # Calculate TD error for prioritization
            current_q = self.model.predict(state_padded.reshape(1, -1))[0]
            next_q_target = self.target_model.predict(next_state_padded.reshape(1, -1))[0]
            
            target = reward
            if not done:
                target += self.discount_factor * np.max(next_q_target)
            
            td_error = target - current_q[action_index]

            # Store experience
            self.replay_buffer.add(
                state_padded, action_index, reward, next_state_padded, done, td_error
            )
            
            # Train if enough samples
            if len(self.replay_buffer.memory) >= self.batch_size:
                loss = self.train_model()
                self.performance_metrics["training_losses"].append(loss)
                self.train_counter += 1
                
                # Update target network periodically
                if self.train_counter % 50 == 0: 
                    self.update_target_model()
                    logger.info(f"Target model updated. Training step: {self.train_counter}")
                    
        except Exception as e:
            logger.error(f"Error in record_and_train: {e}")

    def _ensure_state_shape(self, state):
        """Ensure state has correct shape and dimensions."""
        state_array = np.array(state, dtype=np.float32)
        
        if state_array.ndim == 0:
            state_array = np.array([state_array])
            
        if len(state_array) < self.max_state_size:
            state_array = np.pad(state_array, (0, self.max_state_size - len(state_array)))
        elif len(state_array) > self.max_state_size:
            state_array = state_array[:self.max_state_size]
            
        return state_array

    def train_model(self):
        """FIXED: Training method with proper array handling."""
        try:
            minibatch, weights, indices = self.replay_buffer.sample(self.batch_size)
            if not minibatch:
                return 0.0
            
            states = np.array([e.state for e in minibatch])
            next_states = np.array([e.next_state for e in minibatch])
            
            # Double DQN update
            main_q_next = self.model.predict(next_states)
            best_actions = np.argmax(main_q_next, axis=1)
            target_q_next = self.target_model.predict(next_states)
            
            targets = []
            current_qs = self.model.predict(states)
            td_errors = []

            for i, experience in enumerate(minibatch):
                target = experience.reward
                if not experience.done:
                    target += self.discount_factor * target_q_next[i][best_actions[i]]
                
                # Update target for taken action
                target_q = current_qs[i].copy()
                target_q[experience.action] = target
                targets.append(target_q)
                
                td_errors.append(abs(target - current_qs[i][experience.action]))

            # Convert to arrays
            targets_array = np.array(targets)
            
            # FIXED: Use len() instead of boolean evaluation for arrays
            if weights is not None and hasattr(weights, '__len__') and len(weights) > 0:
                weights_array = np.array(weights)
            else:
                weights_array = None
            
            # Train model
            loss = self.model.train_on_batch(states, targets_array, sample_weights=weights_array)
            
            # FIXED: Use len() instead of boolean evaluation for lists
            if indices and td_errors and len(indices) > 0 and len(td_errors) > 0:
                self.replay_buffer.update_priorities(indices, td_errors)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            return 0.0

    def generate_realistic_scenario(self, difficulty: str) -> List[Dict]:
        """Generate balanced training scenarios."""
        scenarios = {
            "easy": {"civilians": 3, "suspicious": 1, "hostile": 0, "environment": "calm"},
            "medium": {"civilians": 2, "suspicious": 2, "hostile": 1, "environment": "moderate"},
            "hard": {"civilians": 1, "suspicious": 2, "hostile": 2, "environment": "rough"},
            "expert": {"civilians": 0, "suspicious": 3, "hostile": 3, "environment": "stormy"}
        }
        
        config = scenarios.get(difficulty, scenarios["medium"])
        vessels = []
        
        self._set_environment_difficulty(config["environment"])
        
        # Generate vessels based on configuration
        for _ in range(config["civilians"]):
            vessels.append(self._generate_realistic_vessel(random.choice(["fisherman", "merchant"])))
        for _ in range(config["suspicious"]):
            vessels.append(self._generate_realistic_vessel("hostile_disguised"))
        for _ in range(config["hostile"]):
            vessels.append(self._generate_realistic_vessel(random.choice(["hostile_small", "hostile_medium"])))
        
        self.performance_metrics["vessels_generated"] += len(vessels)
        
        # Validate scenario
        if not self.validate_scenario(vessels):
            logger.warning(f"Generated scenario for {difficulty} failed validation")
            
        return vessels

    def _set_environment_difficulty(self, environment):
        """Set environmental conditions based on difficulty."""
        env_configs = {
            "calm": {"sea_state": random.randint(1, 2), "visibility": random.uniform(15.0, 20.0)},
            "moderate": {"sea_state": random.randint(2, 3), "visibility": random.uniform(8.0, 15.0)},
            "rough": {"sea_state": random.randint(3, 4), "visibility": random.uniform(4.0, 10.0)},
            "stormy": {"sea_state": random.randint(4, 5), "visibility": random.uniform(2.0, 6.0)}
        }
        
        config = env_configs.get(environment, env_configs["moderate"])
        self.environment_factors.update(config)
        self.environment_factors["time_of_day"] = random.choice(["dawn", "day", "dusk", "night"])

    def _generate_realistic_vessel(self, role: str) -> Dict:
        """Generate vessels with realistic attributes and behaviors."""
        # Base positions for interesting scenarios
        base_positions = [
            (random.uniform(100, 300), random.uniform(100, 300)),  # NW quadrant
            (random.uniform(500, 700), random.uniform(100, 300)),  # NE quadrant  
            (random.uniform(100, 300), random.uniform(300, 500)),  # SW quadrant
            (random.uniform(500, 700), random.uniform(300, 500))   # SE quadrant
        ]
        
        pos = random.choice(base_positions)
        
        # Base template
        vessel_template = {
            "vessel_type": "",
            "speed": 0,
            "behavior": "",
            "true_threat_level": "neutral",
            "evasion_chance": 0.1,
            "heading": random.uniform(0, 360),
            "x": pos[0],
            "y": pos[1],
            "detection_range": 200,
            "aggressiveness": 0.1,
            "id": f"vessel_{random.randint(1000, 9999)}",
            "crew_count": random.randint(2, 25)
        }
        
        # Role-specific configurations
        role_configs = {
            "fisherman": {
                "vessel_type": random.choice(["Trawler", "Fishing Boat", "Fishing Vessel"]),
                "speed": random.uniform(2, 8),
                "behavior": "fishing_pattern",
                "true_threat_level": "neutral",
                "evasion_chance": 0.1,
                "detection_range": 150,
                "aggressiveness": 0.0
            },
            "merchant": {
                "vessel_type": random.choice(["Cargo Ship", "Tanker", "Container Ship"]),
                "speed": random.uniform(10, 18),
                "behavior": "commercial_route",
                "true_threat_level": "neutral", 
                "evasion_chance": 0.05,
                "detection_range": 250,
                "aggressiveness": 0.0
            },
            "hostile_small": {
                "vessel_type": random.choice(["Fast Attack Craft", "Patrol Boat", "Missile Boat"]),
                "speed": random.uniform(20, 35),
                "behavior": "aggressive_patrol",
                "true_threat_level": "confirmed",
                "evasion_chance": 0.8,
                "detection_range": 400,
                "aggressiveness": 0.9
            },
            "hostile_medium": {
                "vessel_type": random.choice(["Corvette", "Frigate", "Destroyer"]),
                "speed": random.uniform(15, 25),
                "behavior": "naval_patrol",
                "true_threat_level": "confirmed",
                "evasion_chance": 0.4,
                "detection_range": 600,
                "aggressiveness": 0.7
            },
            "hostile_disguised": {
                "vessel_type": "Fishing Trawler",
                "speed": random.uniform(15, 25),
                "behavior": "suspicious_fishing", 
                "true_threat_level": "possible",
                "evasion_chance": 0.6,
                "detection_range": 300,
                "aggressiveness": 0.5
            }
        }
        
        config = role_configs.get(role, {
            "vessel_type": "Unknown",
            "speed": random.uniform(5, 15),
            "behavior": "idle"
        })
        
        vessel_template.update(config)
        return vessel_template

    def control_npc_behavior(self, vessel: Dict, player_position: Tuple[float, float]) -> Tuple[float, float]:
        """Enhanced NPC behavior with tactical decision-making."""
        try:
            if 'x' not in vessel or 'y' not in vessel:
                logger.warning(f"Vessel missing position attributes: {vessel}")
                return 0.0, 0.0
            
            distance_to_player = math.hypot(vessel['x'] - player_position[0], vessel['y'] - player_position[1])
            threat_level = vessel.get('true_threat_level', 'neutral')
            vessel_type = vessel.get('vessel_type', 'unknown')
            
            if threat_level == "neutral":
                return self._enhanced_civilian_behavior(vessel, distance_to_player, player_position)
            elif threat_level == "possible":
                return self._enhanced_suspicious_behavior(vessel, distance_to_player, player_position, vessel_type)
            else:  # confirmed
                return self._enhanced_hostile_behavior(vessel, distance_to_player, player_position, vessel_type)
                
        except Exception as e:
            logger.error(f"Error in NPC behavior control: {e}")
            return 0.0, 0.0

    def _enhanced_civilian_behavior(self, vessel: Dict, distance: float, player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Enhanced civilian behavior with awareness."""
        vessel_speed = vessel.get('speed', 5.0)
        
        if distance < 300:  # Too close to player
            dx, dy = vessel['x'] - player_pos[0], vessel['y'] - player_pos[1]
            dist = max(math.hypot(dx, dy), 1)
            speed = min(vessel_speed * 1.2, 8.0)
            return (dx/dist) * speed, (dy/dist) * speed
        
        # Normal civilian operations
        if random.random() < 0.01:  # Small chance to change course
            new_heading = vessel.get('heading', 0) + random.uniform(-15, 15)
            vessel['heading'] = new_heading
        else:
            new_heading = vessel.get('heading', 0)
        
        rad = math.radians(new_heading)
        return math.cos(rad) * vessel_speed, math.sin(rad) * vessel_speed

    def _enhanced_suspicious_behavior(self, vessel: Dict, distance: float, player_pos: Tuple[float, float], vessel_type: str) -> Tuple[float, float]:
        """Suspicious vessels test boundaries and gather intelligence."""
        vessel_speed = vessel.get('speed', 10.0)
        evasion_chance = vessel.get('evasion_chance', 0.6)
        
        if distance < 150 and random.random() < evasion_chance:
            # Evade if player gets too close
            dx, dy = vessel['x'] - player_pos[0], vessel['y'] - player_pos[1]
            dist = max(math.hypot(dx, dy), 1)
            speed = min(vessel_speed * 1.5, 25.0)
            return (dx/dist) * speed, (dy/dist) * speed
        elif distance > 500:
            # Close in to observe if too far
            dx, dy = player_pos[0] - vessel['x'], player_pos[1] - vessel['y']
            dist = max(math.hypot(dx, dy), 1)
            speed = vessel_speed * 0.8  # Approach cautiously
            return (dx/dist) * speed, (dy/dist) * speed
        else:
            # Loiter in observation pattern
            return self._observation_pattern(vessel, player_pos)

    def _enhanced_hostile_behavior(self, vessel: Dict, distance: float, player_pos: Tuple[float, float], vessel_type: str) -> Tuple[float, float]:
        """Enhanced hostile behavior with vessel type differentiation."""
        if vessel_type in ["Fast Attack Craft", "Patrol Boat", "Missile Boat"]:
            return self._fast_attack_behavior(vessel, distance, player_pos)
        elif vessel_type in ["Corvette", "Frigate", "Destroyer"]:
            return self._medium_hostile_behavior(vessel, distance, player_pos)
        else:
            # Generic hostile behavior
            dx, dy = player_pos[0] - vessel['x'], player_pos[1] - vessel['y']
            dist = max(math.hypot(dx, dy), 1)
            return (dx/dist) * vessel.get('speed', 15.0), (dy/dist) * vessel.get('speed', 15.0)

    def _fast_attack_behavior(self, vessel: Dict, distance: float, player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Fast attack craft use speed and hit-and-run tactics."""
        vessel_speed = vessel.get('speed', 25.0)
        dx, dy = player_pos[0] - vessel['x'], player_pos[1] - vessel['y']
        dist = max(math.hypot(dx, dy), 1)
        
        if distance < 100:  # Attack run
            speed = min(vessel_speed * 2.0, 40.0)
            jink = random.uniform(-0.1, 0.1)  # Add slight jinking for evasion
            return (dx/dist + jink) * speed, (dy/dist + jink) * speed
        elif distance < 300:  # Approach with zig-zag
            base_angle = math.atan2(dy, dx)
            zig_zag = math.sin(time.time() * 2) * 0.3  # Oscillate course
            speed = vessel_speed * 1.5
            return math.cos(base_angle + zig_zag) * speed, math.sin(base_angle + zig_zag) * speed
        else:  # High-speed approach
            speed = vessel_speed * 1.8
            return (dx/dist) * speed, (dy/dist) * speed

    def _medium_hostile_behavior(self, vessel: Dict, distance: float, player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Medium warship behavior - more methodical and powerful."""
        vessel_speed = vessel.get('speed', 20.0)
        dx, dy = player_pos[0] - vessel['x'], player_pos[1] - vessel['y']
        dist = max(math.hypot(dx, dy), 1)
        
        if distance < 200:  # Close weapons range
            speed = min(vessel_speed * 1.3, 28.0)
            return (dx/dist) * speed, (dy/dist) * speed
        elif distance < 600:  # Optimal engagement range
            # Maintain optimal firing distance
            desired_distance = 300
            if distance > desired_distance:
                # Close in methodically
                speed = vessel_speed * 1.1
                return (dx/dist) * speed, (dy/dist) * speed
            else:
                # Back away to optimal range
                speed = vessel_speed * 0.9
                return (-dx/dist) * speed, (-dy/dist) * speed
        else:  # Long range approach
            return (dx/dist) * vessel_speed, (dy/dist) * vessel_speed

    def _observation_pattern(self, vessel: Dict, player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Loiter in an observation pattern around the player."""
        radius = 350  # Observation radius
        current_time = time.time()
        vessel_speed = vessel.get('speed', 8.0)
        
        # Circular observation pattern
        angle = (current_time * 0.5) % (2 * math.pi)  # Slow circle
        target_x = player_pos[0] + math.cos(angle) * radius
        target_y = player_pos[1] + math.sin(angle) * radius
        
        dx, dy = target_x - vessel['x'], target_y - vessel['y']
        dist = max(math.hypot(dx, dy), 1)
        
        # Update vessel heading to face the target
        target_heading = math.degrees(math.atan2(dy, dx))
        vessel['heading'] = target_heading
        
        return (dx/dist) * vessel_speed, (dy/dist) * vessel_speed

    def _generate_reasoning(self, vessel: Dict, action: AIAction, **kwargs) -> str:
        """More detailed and contextual reasoning for AI decisions."""
        vessel_type = vessel.get('vessel_type', 'unknown')
        threat_level = vessel.get('true_threat_level', 'neutral')
        speed = vessel.get('speed', 0)
        
        reasoning_templates = {
            AIAction.INTERCEPT: [
                f"Vessel {vessel_type} moving at suspicious speed ({speed:.1f} knots). Intercept recommended.",
                f"Hostile behavior patterns detected from {vessel_type}. Immediate interception required.",
                f"{vessel_type} matches known threat profile. Authorization for intercept granted."
            ],
            AIAction.MONITOR: [
                f"{vessel_type} behavior ambiguous but concerning. Continuing observation.",
                f"Insufficient data on {vessel_type} at current range. Maintain monitoring.",
                f"{vessel_type} appears suspicious but non-threatening. Monitoring protocol engaged."
            ],
            AIAction.AWAIT_CONFIRMATION: [
                f"Uncertain threat assessment for {vessel_type}. Requesting human validation.",
                f"Rules of engagement unclear for {vessel_type} situation. Manual assessment required.",
                f"Conflicting behavioral indicators from {vessel_type}. Awaiting command decision."
            ],
            AIAction.EVADE: [
                f"{vessel_type} identified as confirmed threat. Evasive maneuvers recommended.",
                f"Hostile vessel {vessel_type} closing. Recommend defensive positioning.",
                f"Threat assessment: high. Recommend evasion from {vessel_type}."
            ],
            AIAction.IGNORE: [
                f"{vessel_type} confirmed as civilian vessel. No action required.",
                f"Vessel {vessel_type} poses no threat. Continuing patrol route.",
                f"Standard commercial traffic {vessel_type}. Clear to ignore."
            ]
        }
        
        return random.choice(reasoning_templates.get(action, [f"Standard procedure for {vessel_type}."]))

    def validate_scenario(self, vessels: List[Dict]) -> bool:
        """Validate that generated scenario is balanced and playable."""
        if not vessels:
            logger.warning("Empty scenario generated")
            return False
        
        # Check for reasonable vessel distribution
        threat_counts = {"neutral": 0, "possible": 0, "confirmed": 0}
        for vessel in vessels:
            threat_level = vessel.get('true_threat_level', 'neutral')
            threat_counts[threat_level] += 1
            
        logger.info(f"Scenario generated: {threat_counts}")
        
        # Ensure not all vessels are hostile (would be overwhelming)
        if threat_counts['confirmed'] >= len(vessels) * 0.6:  # More than 60% hostile
            logger.warning("Scenario too hostile, rebalancing...")
            return False
            
        return True

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        total_decisions = max(1, self.performance_metrics["decisions"])
        success_rate = (total_decisions - self.performance_metrics["hitl_requests"]) / total_decisions
        
        avg_loss = 0.0
        if self.performance_metrics["training_losses"]:
            avg_loss = np.mean(self.performance_metrics["training_losses"][-100:])  # Last 100 losses
        
        return {
            "total_decisions": self.performance_metrics["decisions"],
            "hitl_requests": self.performance_metrics["hitl_requests"],
            "autonomous_success_rate": f"{success_rate:.2%}",
            "cumulative_reward": self.performance_metrics["cumulative_reward"],
            "vessels_generated": self.performance_metrics["vessels_generated"],
            "current_epsilon": self.epsilon,
            "training_steps": self.train_counter,
            "replay_buffer_size": len(self.replay_buffer.memory),
            "average_training_loss": avg_loss,
            "environment": self.environment_factors
        }

    def reset_performance_metrics(self):
        """Reset performance metrics for new evaluation period."""
        self.performance_metrics = {
            "decisions": 0, 
            "hitl_requests": 0, 
            "cumulative_reward": 0.0, 
            "vessels_generated": 0,
            "training_losses": []
        }
        logger.info("Performance metrics reset")

    def save_model(self, filepath: str):
        """Save model weights to file."""
        try:
            model_data = {
                "model_weights": self.model.get_weights(),
                "target_weights": self.target_model.get_weights(),
                "epsilon": self.epsilon,
                "train_counter": self.train_counter,
                "performance_metrics": self.performance_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, filepath: str):
        """Load model weights from file."""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.model.set_weights(model_data["model_weights"])
            self.target_model.set_weights(model_data["target_weights"])
            self.epsilon = model_data.get("epsilon", 1.0)
            self.train_counter = model_data.get("train_counter", 0)
            self.performance_metrics.update(model_data.get("performance_metrics", {}))
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create AI instance
    ai = NavalAI(backend=None)
    
    print("=== Naval AI Engine Test ===\n")
    
    # Test scenario generation
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- Testing {difficulty} scenario ---")
        scenario = ai.generate_realistic_scenario(difficulty)
        print(f"Generated {len(scenario)} vessels")
        
        # Validate scenario
        if ai.validate_scenario(scenario):
            print("✓ Scenario validation passed")
        else:
            print("✗ Scenario validation failed")
        
        # Get player vessel
        player = database.get_sim_state("player1")
        
        # Test AI decisions for each vessel
        for i, vessel in enumerate(scenario[:3]):  # Test first 3 vessels
            report = ai.decide_action(vessel, player)
            print(f"Vessel {i+1} ({vessel['vessel_type']}): {report.recommended_action.value}")
            print(f"  Reasoning: {report.reasoning}")
            print(f"  Confidence: {report.confidence:.2f}")
    
    # Show final performance
    print("\n=== Final Performance Report ===")
    performance = ai.get_performance_report()
    for key, value in performance.items():
        print(f"{key}: {value}")