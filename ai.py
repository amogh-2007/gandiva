# ai.py
"""
Naval Combat AI Engine with Advanced Reinforcement Learning
Implements Double Deep Q-Learning (DDQN), Prioritized Experience Replay,
a formal Human-in-the-Loop (HITL) framework, and stubs for adversarial defense
and multi-agent coordination.
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
import logging  # Added for better error handling

# Professional logging and error resilience
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock database module (replace with real one for production)
class MockDatabase:
    def get_boat(self, vessel_id):
        # Simple mock: returns a dict with vessel data
        return {"x": random.uniform(100, 700), "y": random.uniform(100, 500),
                "speed": random.uniform(0, 10), "threat_level": random.choice(["neutral", "possible", "confirmed"])}
    
    def get_sim_state(self, player_id):
        return {"player_x": 400, "player_y": 300}  # Fixed player position for demo

database = MockDatabase()  # Use mock for demonstration

# --- Placeholder for a real Deep Learning Framework ---
class DQNetwork:
    """Placeholder for a Deep Q-Network. In a real implementation, this would be a Keras/PyTorch model."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.rand(10)  # Simulate weights
        logger.info(f"Initialized mock DQN with state size {state_size} and action size {action_size}")

    def predict(self, state):
        """Simulates predicting Q-values for all actions from a state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        # Robust State Management: Pad or truncate input to expected size
        if state.shape[1] < self.state_size:
            padding = np.zeros((state.shape[0], self.state_size - state.shape[1]))
            state = np.hstack([state, padding])
        elif state.shape[1] > self.state_size:
            state = state[:, :self.state_size]
        return np.random.rand(state.shape[0], self.action_size)  # Mock Q-values

    def train_on_batch(self, states, targets, sample_weights=None):
        """Simulates training with support for sample weights from PER."""
        loss = np.mean((self.predict(states) - targets) ** 2)
        if sample_weights is not None:
            loss = np.mean(sample_weights * (self.predict(states) - targets) ** 2)
        logger.info(f"Mock training loss: {loss:.4f}")
        self.weights += np.random.normal(0, 0.01, self.weights.shape)
    
    def get_weights(self):
        """Simulates getting model weights."""
        return {"sim_weights": self.weights.tolist()}

    def set_weights(self, weights):
        """Simulates setting model weights."""
        if "sim_weights" in weights:
            self.weights = np.array(weights["sim_weights"])

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    """Enhanced PER implementation with importance sampling weights."""
    def __init__(self, buffer_size=10000, alpha=0.6, beta=0.4):  # Added beta for importance sampling
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.td_errors = deque(maxlen=buffer_size)  # Track errors for updates

    def add(self, state, action, reward, next_state, done, error):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
        td_error = abs(error) + 1e-5  # Small epsilon to avoid zero
        priority = td_error ** self.alpha
        self.priorities.append(priority)
        self.td_errors.append(td_error)

    def sample(self, batch_size=64):
        """Samples a batch, returning experiences, weights, and indices."""
        if not self.memory:
            return [], np.ones(batch_size), []  # No weights if empty
        prios = np.array(self.priorities)
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.memory), min(len(self.memory), batch_size), p=probs)
        samples = [self.memory[i] for i in indices]
        # Importance Sampling Weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, errors):
        """Update priorities after training (basic PER feature)."""
        for idx, error in zip(indices, errors):
            if idx < len(self.td_errors):
                self.td_errors[idx] = abs(error) + 1e-5
                self.priorities[idx] = (self.td_errors[idx]) ** self.alpha
# --- End of Placeholder Section ---

class ThreatLevel(Enum):
    NEUTRAL = "neutral"; POSSIBLE = "possible"; CONFIRMED = "confirmed"

class AIAction(Enum):
    INTERCEPT = "intercept"; MONITOR = "monitor"; IGNORE = "ignore"; EVADE = "evade"; AWAIT_CONFIRMATION = "await_confirmation"

@dataclass
class AIReport:
    """Standardized AI status report."""
    vessel_id: str; threat_assessment: str; recommended_action: AIAction; confidence: float; reasoning: str; timestamp: float

class NavalAI:
    def __init__(self, backend, state_size=5, squad_state_size=3, action_size=len(AIAction)):
        self.backend = backend
        self.vessel_templates = self._load_vessel_templates()

        # State and Model components
        self.base_state_size = state_size
        self.squad_state_size = squad_state_size  # Added for MARL
        self.max_state_size = state_size + squad_state_size  # Fixed max for model
        self.action_size = action_size
        self.model = DQNetwork(self.max_state_size, action_size)
        self.target_model = DQNetwork(self.max_state_size, action_size)
        self.update_target_model()

        # RL components
        self.replay_buffer = PrioritizedReplayBuffer()
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.train_counter = 0

        # HITL & Curriculum Learning
        self.hitl_confidence_threshold = 0.7
        self.reward_shaping = {"human_override": -15, "human_confirm": +10}
        self.difficulty = "easy"

        # Monitoring & Adversarial Framework
        self.performance_metrics = {"decisions": 0, "hitl_requests": 0, "cumulative_reward": 0.0, "adversarial_success_rate": 0.0}
        self.adversarial_training_ratio = 0.15

        # MARL Stubs
        self.squad_id = "alpha_squad"
        self.squad_members = []  # List of other agent IDs

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_state(self, vessel_id: str) -> np.ndarray:
        try:
            vessel = database.get_boat(vessel_id)
            player_state = database.get_sim_state(1)
            player_pos = np.array([player_state.get("player_x", 0), player_state.get("player_y", 0)])
        except Exception as e:
            logger.warning(f"Error fetching state for {vessel_id}: {e}")
            return np.zeros(self.base_state_size)  # Graceful degradation
        if not vessel:
            return np.zeros(self.base_state_size)
        vessel_pos = np.array([vessel.get('x', 0), vessel.get('y', 0)])
        relative_pos = vessel_pos - player_pos
        return np.array([
            np.linalg.norm(relative_pos) / 1000.0,
            vessel.get('speed', 0) / 10.0,
            math.atan2(relative_pos[1], relative_pos[0]) / math.pi,
            1 if vessel.get('threat_level') == 'possible' else 0,
            1 if vessel.get('threat_level') == 'confirmed' else 0
        ])

    def _process_squad_observations(self, squad_observations: Dict) -> np.ndarray:
        """Added: Process squad data into a fixed-size vector (e.g., avg threat, count, coordination signal)."""
        if not squad_observations:
            return np.zeros(self.squad_state_size)
        # Example: [avg_threat (0-1), num_active_squad (int), coordination_factor (0-1)]
        threats = [obs.get('threat', 0) for obs in squad_observations.values()]
        avg_threat = np.mean(threats) if threats else 0
        num_active = len([t for t in threats if t > 0])
        coord_factor = min(1.0, num_active / len(squad_observations)) if squad_observations else 0
        return np.array([avg_threat, num_active, coord_factor])

    def decide_action(self, vessel_id, squad_observations: Optional[Dict] = None) -> AIReport:
        """Makes a decision using epsilon-greedy DQN, HITL, and MARL context."""
        state = self.get_state(vessel_id)
        # MARL: Augment state with squad observations
        squad_vec = self._process_squad_observations(squad_observations) if squad_observations else np.zeros(self.squad_state_size)
        state = np.append(state, squad_vec)
        
        state_reshaped = state.reshape(1, -1)
        
        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(self.action_size)
            q_values = np.random.rand(self.action_size)  # Mock for confidence
            confidence = 0.5
        else:
            q_values = self.model.predict(state_reshaped)[0]
            action_index = np.argmax(q_values)
            # Fixed: Proper softmax for confidence
            exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
            confidence = float(exp_q[action_index] / np.sum(exp_q))
        
        action = list(AIAction)[action_index]
        vessel_data = database.get_boat(vessel_id)

        if confidence < self.hitl_confidence_threshold and action == AIAction.INTERCEPT:
            action = AIAction.AWAIT_CONFIRMATION
            self.performance_metrics["hitl_requests"] += 1
            reasoning = f"AI low confidence ({confidence:.2f}). Please validate INTERCEPT."
        else:
            reasoning = self._generate_reasoning(vessel_data, action)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.performance_metrics["decisions"] += 1
        return AIReport(vessel_id, "dynamic", action, confidence, reasoning, time.time())

    def record_and_train(self, state, action_index, reward, next_state, done, human_feedback=None):
        if human_feedback == "override":
            reward += self.reward_shaping["human_override"]
        elif human_feedback == "confirm":
            reward += self.reward_shaping["human_confirm"]

        # Pad states for consistency before calculating error and storing
        state_padded = np.pad(state, (0, self.max_state_size - len(state))) if len(state) < self.max_state_size else state
        next_state_padded = np.pad(next_state, (0, self.max_state_size - len(next_state))) if len(next_state) < self.max_state_size else next_state

        current_q = self.model.predict(state_padded.reshape(1, -1))[0]
        next_q_target = self.target_model.predict(next_state_padded.reshape(1, -1))[0]
        target = reward + self.discount_factor * np.amax(next_q_target) if not done else reward
        error = target - current_q[action_index]

        self.replay_buffer.add(state_padded, action_index, reward, next_state_padded, done, error)
        self.performance_metrics["cumulative_reward"] += reward
        
        if len(self.replay_buffer.memory) > self.batch_size:
            self.train_model()
            self.train_counter += 1
            if self.train_counter % 10 == 0:
                self.update_target_model()

    def train_model(self):
        minibatch, weights, indices = self.replay_buffer.sample(self.batch_size)
        if not minibatch:
            return
        
        states = np.array([e.state for e in minibatch])
        next_states = np.array([e.next_state for e in minibatch])
        
        main_q_next = self.model.predict(next_states)
        best_actions = np.argmax(main_q_next, axis=1)
        target_q_next = self.target_model.predict(next_states)
        
        targets = []
        current_qs = self.model.predict(states)
        td_errors = []

        for i, experience in enumerate(minibatch):
            target = experience.reward
            if not experience.done:
                target = experience.reward + self.discount_factor * target_q_next[i][best_actions[i]]
            
            q_values = current_qs[i].copy()
            q_values[experience.action] = target
            targets.append(q_values)
            td_errors.append(abs(target - current_qs[i][experience.action]))  # For priority update

        self.model.train_on_batch(states, np.array(targets), sample_weights=weights)
        
        # Update priorities (added for better PER)
        self.replay_buffer.update_priorities(indices, td_errors)
        
        if random.random() < self.adversarial_training_ratio:
            self.run_adversarial_training_step()

    # --- 1. Advanced Adversarial Attacks ---
    def _generate_fgsm_attack(self, state, epsilon=0.02):
        """Missing: Fast Gradient Sign Method (FGSM)"""
        logger.info("Called stub for FGSM attack.")
        return np.clip(state + epsilon * np.sign(np.random.normal(size=state.shape)), 0, 1)

    def _generate_pgd_attack(self, state, epsilon=0.02, steps=5):
        """Missing: Projected Gradient Descent (PGD)"""
        logger.info("Called stub for PGD attack.")
        adv_state = state.copy()
        for _ in range(steps):
            perturbation = (epsilon / steps) * np.sign(np.random.normal(size=adv_state.shape))
            adv_state = np.clip(adv_state + perturbation, state - epsilon, state + epsilon)
        return adv_state

    def run_adversarial_training_step(self):
        if not self.replay_buffer.memory: return
        real_state = self.replay_buffer.memory[-1].state
        adversarial_state = (self._generate_pgd_attack(real_state) if random.random() > 0.5 else self._generate_fgsm_attack(real_state))
        
        target_q_values = self.target_model.predict(np.reshape(real_state, [1, -1]))
        self.model.train_on_batch(np.array([adversarial_state]), target_q_values)

    # --- 2. Multi-Agent Coordination Execution ---
    def get_squad_action(self, agent_id, all_squad_states: Dict):
        """Missing: Actual multi-agent decision making"""
        logger.info(f"Agent {agent_id} getting squad action (stub).")
        return self.decide_action(agent_id, all_squad_states)

    def share_observations(self, squad_members):
        """Missing: Communication protocol between agents"""
        logger.info(f"Sharing observations among {len(squad_members)} members (stub).")
        return {} # Placeholder

    # --- 3. Comprehensive Monitoring & Evaluation ---
    def evaluate_adversarial_robustness(self, test_states):
        """Missing: Adversarial success rate tracking"""
        logger.info("Evaluating adversarial robustness (stub).")
        return 0.0

    def explain_decision(self, state, action):
        """Missing: Model explainability tools"""
        logger.info("Generating decision explanation (stub).")
        return "Decision based on: mock feature importance"

    # --- 4. Formal Curriculum Learning ---
    def update_curriculum(self, player_success_rate):
        """Missing: Difficulty scheduling based on player performance"""
        if player_success_rate > 0.8 and self.difficulty == "easy":
            self.difficulty = "medium"
            logger.info("Curriculum updated: Difficulty increased to MEDIUM.")
        elif player_success_rate < 0.4 and self.difficulty == "medium":
            self.difficulty = "easy"
            logger.info("Curriculum updated: Difficulty decreased to EASY.")

    # --- Other helper methods ---
    def _load_vessel_templates(self): return {}
    def generate_vessels(self, count, mission_type, adversarial_mode=False): pass
    def _select_vessel_template(self, mission_type): return "hostile_vessel"
    def _generate_safe_position(self, existing): return (0,0)
    def _generate_velocity(self, speed_range): return (0,0)
    def _generate_reasoning(self, vessel, action, **kwargs): return f"Action: {action.name}"

