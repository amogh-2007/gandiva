# ai_training.py
"""
Enhanced AI Training Pipeline for Naval Combat Simulation
Optimized and cleaned version with reduced redundancy
"""

import time
import numpy as np
from backend import SimulationController, Vessel
from ai import NavalAI, AIAction, AIReport
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedRewardCalculator:
    """Balanced reward calculation to address missed threats"""
    
    @staticmethod
    def calculate_reward(ai_report: AIReport, vessel, action_taken: str) -> float:
        """Calculate reward with balanced threat detection incentives"""
        try:
            if hasattr(vessel, 'true_threat_level'):
                true_threat = vessel.true_threat_level
            else:
                true_threat = vessel.get('true_threat_level', 'neutral')
            
            reward = FixedRewardCalculator._get_base_reward(true_threat, ai_report.recommended_action)
            reward = FixedRewardCalculator._apply_confidence_adjustments(reward, ai_report.confidence)
            
            logger.debug(f"Reward: {true_threat} + {ai_report.recommended_action.value} = {reward:.2f}")
            return reward
            
        except Exception as e:
            logger.error(f"Error in reward calculation: {e}")
            return 0.0
    
    @staticmethod
    def _get_base_reward(true_threat: str, action: AIAction) -> float:
        """Get base reward for threat-action combination"""
        reward_matrix = {
            "confirmed": {
                AIAction.INTERCEPT: 25.0,
                AIAction.MONITOR: -10.0,
                AIAction.IGNORE: -35.0,
                AIAction.AWAIT_CONFIRMATION: -5.0,
                AIAction.EVADE: 8.0
            },
            "possible": {
                AIAction.INTERCEPT: 8.0,
                AIAction.MONITOR: 15.0,
                AIAction.IGNORE: -15.0,
                AIAction.AWAIT_CONFIRMATION: 8.0,
                AIAction.EVADE: 10.0
            },
            "neutral": {
                AIAction.INTERCEPT: -20.0,
                AIAction.MONITOR: 1.0,
                AIAction.IGNORE: 6.0,
                AIAction.AWAIT_CONFIRMATION: -3.0,
                AIAction.EVADE: -8.0
            }
        }
        
        return reward_matrix.get(true_threat, {}).get(action, -2.0)
    
    @staticmethod
    def _apply_confidence_adjustments(reward: float, confidence: float) -> float:
        """Apply confidence-based adjustments"""
        if confidence > 0.8:
            return reward + 5.0
        elif confidence < 0.3:
            return reward - 4.0
        return reward

class EnhancedAITrainingPipeline:
    """Optimized training pipeline with improved threat handling"""
    
    def __init__(self, episodes=100, steps_per_episode=50, enable_debug=False):
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.enable_debug = enable_debug
        self.simulation = SimulationController(mission_type="Training", difficulty="medium")
        
        # Initialize AI with optimized parameters
        self.ai = NavalAI(backend=None)
        self.ai.hitl_confidence_threshold = 0.6
        self.ai.epsilon_decay = 0.995
        
        self._ensure_ai_attributes()
        self.reward_calculator = FixedRewardCalculator()
        
        # Consolidated training metrics
        self.training_metrics = self._initialize_metrics()
        
        if self.enable_debug:
            logger.setLevel(logging.DEBUG)
            print("🔧 Debug mode enabled")

    def _initialize_metrics(self):
        """Initialize training metrics structure"""
        return {
            'episode_rewards': [],
            'episode_threat_accuracy': [],
            'action_counts': {
                'successful_intercepts': 0,
                'false_positives': 0, 
                'missed_threats': 0,
                'correct_monitors': 0,
                'correct_ignores': 0,
                'hitl_requests': 0
            },
            'confidence_stats': {'high': 0, 'medium': 0, 'low': 0},
            'threat_type_breakdown': {'confirmed': 0, 'possible': 0, 'neutral': 0}
        }

    def _ensure_ai_attributes(self):
        """Ensure AI has all required attributes"""
        if not hasattr(self.ai, 'train_counter'):
            self.ai.train_counter = 0
        if not hasattr(self.ai, 'performance_metrics'):
            self.ai.performance_metrics = {
                "decisions": 0, "hitl_requests": 0, "cumulative_reward": 0.0,
                "vessels_generated": 0, "training_losses": []
            }

    def run_comprehensive_test(self):
        """Run comprehensive system test before training"""
        print("\n🔍 COMPREHENSIVE SYSTEM TEST")
        print("=" * 50)
        
        test_results = {
            'AI Initialization': self.test_ai_initialization(),
            'Scenario Generation': self.test_scenario_generation(),
            'Reward Calculation': self.test_reward_calculation(),
            'Training Integration': self.test_training_integration()
        }
        
        tests_passed = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! Ready for training.")
            return True
        else:
            print("⚠️  Some tests failed. Check the issues above.")
            return False

    def test_ai_initialization(self):
        """Test AI initialization and basic functionality"""
        try:
            assert hasattr(self.ai, 'model'), "AI model not initialized"
            assert hasattr(self.ai, 'replay_buffer'), "Replay buffer not initialized"
            
            test_vessel = self._create_test_vessel("Test Boat", "neutral", 300, 300)
            test_player = self._create_test_player()
            report = self.ai.decide_action(test_vessel, test_player)
            
            assert isinstance(report, AIReport), "AI should return AIReport"
            assert report.confidence >= 0, "Confidence should be non-negative"
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def test_scenario_generation(self):
        """Test scenario generation and vessel conversion"""
        try:
            for difficulty in ["easy", "medium", "hard"]:
                scenario = self.ai.generate_realistic_scenario(difficulty)
                assert len(scenario) > 0, f"No vessels generated for {difficulty}"
                
                # Quick threat distribution check
                threats = {"neutral": 0, "possible": 0, "confirmed": 0}
                for vessel in scenario:
                    threat_level = vessel.get('true_threat_level', 'neutral')
                    threats[threat_level] += 1
                print(f"   {difficulty}: {threats}")
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def test_reward_calculation(self):
        """Test reward calculation with corrected expected values"""
        try:
            test_cases = [
                ("confirmed", AIAction.INTERCEPT, 15, "Intercept confirmed threat"),
                ("confirmed", AIAction.IGNORE, -35, "Ignore confirmed threat"),
                ("possible", AIAction.MONITOR, 8, "Monitor possible threat"),
                ("neutral", AIAction.IGNORE, 5, "Ignore neutral vessel"),
            ]
            
            for threat_level, action, expected_min, description in test_cases:
                vessel_data = self._create_test_vessel("Test", threat_level, 300, 300)
                report = AIReport(
                    vessel_id="test", threat_assessment=threat_level,
                    recommended_action=action, confidence=0.7,
                    reasoning="Test", timestamp=time.time()
                )
                
                class MockVessel:
                    def __init__(self, data):
                        self.true_threat_level = data['true_threat_level']
                
                mock_vessel = MockVessel(vessel_data)
                reward = self.reward_calculator.calculate_reward(report, mock_vessel, "test")
                
                if reward >= expected_min:
                    print(f"   ✅ {description}: {reward:.2f}")
                else:
                    print(f"   ❌ {description}: {reward:.2f} (expected >= {expected_min})")
                    return False
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def test_training_integration(self):
        """Test complete training integration"""
        try:
            # Add single test vessel
            vessel_data = self._create_test_vessel("Fishing Boat", "neutral", 200, 200)
            vessel = self.simulation.fleet.add_vessel(
                x=vessel_data['x'], y=vessel_data['y'], vx=0.0, vy=0.0,
                vessel_type=vessel_data['vessel_type'],
                true_threat_level=vessel_data['true_threat_level']
            )
            
            if vessel not in self.simulation.units:
                self.simulation.units.append(vessel)
            
            # Test AI decision and training
            player_data = self._convert_player_to_ai_format()
            vessel_data = self._convert_vessel_to_ai_format({
                'id': vessel.id, 'x': vessel.x, 'y': vessel.y,
                'speed': vessel.speed, 'heading': vessel.heading,
                'vessel_type': vessel.vessel_type,
                'true_threat_level': vessel.true_threat_level
            })
            
            report = self.ai.decide_action(vessel_data, player_data)
            reward = self.reward_calculator.calculate_reward(report, vessel, "test")
            
            # Test training step
            current_state = self.ai.get_state(vessel_data, player_data)
            action_index = list(AIAction).index(report.recommended_action)
            
            self.ai.record_and_train(
                state=current_state, action_index=action_index,
                reward=reward, next_state=current_state,
                done=False, human_feedback=None
            )
            
            print(f"   ✅ Training step: {vessel.vessel_type} -> {report.recommended_action.value}")
            return True
                
        except Exception as e:
            print(f"   Error: {e}")
            return False

    # Helper methods for test data creation
    def _create_test_vessel(self, vessel_type: str, threat_level: str, x: float, y: float) -> dict:
        return {
            'id': f"test_{vessel_type.lower().replace(' ', '_')}",
            'x': x, 'y': y, 'speed': 5.0, 'heading': 0,
            'vessel_type': vessel_type, 'behavior': 'patrol',
            'true_threat_level': threat_level,
            'evasion_chance': 0.7 if threat_level == 'confirmed' else 0.3,
            'detection_range': 200,
            'aggressiveness': 0.8 if threat_level == 'confirmed' else 0.2
        }

    def _create_test_player(self) -> dict:
        return {
            'id': 'player_1', 'x': 500, 'y': 500, 'speed': 0, 'heading': 0,
            'vessel_type': 'Player Ship', 'behavior': 'command',
            'true_threat_level': 'neutral', 'evasion_chance': 0.0,
            'detection_range': 500, 'aggressiveness': 0.0
        }

    def _convert_vessel_to_ai_format(self, vessel_data: dict) -> dict:
        try:
            return {
                'id': vessel_data.get('id', 0), 'x': vessel_data.get('x', 400), 'y': vessel_data.get('y', 300),
                'speed': max(0.1, vessel_data.get('speed', 5.0)), 'heading': vessel_data.get('heading', 0),
                'vessel_type': vessel_data.get('vessel_type', 'Unknown'), 'behavior': vessel_data.get('behavior', 'unknown'),
                'true_threat_level': vessel_data.get('true_threat_level', 'neutral'),
                'evasion_chance': vessel_data.get('evasion_chance', 0.1),
                'detection_range': vessel_data.get('detection_range', 200),
                'aggressiveness': vessel_data.get('aggressiveness', 0.1)
            }
        except Exception as e:
            logger.warning(f"Error converting vessel: {e}")
            return self._create_test_vessel("Unknown", "neutral", 400, 300)

    def _convert_player_to_ai_format(self) -> dict:
        player = self.simulation.player_ship
        return {
            'id': player.id, 'x': player.x, 'y': player.y,
            'speed': max(0.1, player.speed), 'heading': player.heading,
            'vessel_type': 'Player Ship', 'behavior': 'command',
            'true_threat_level': 'neutral', 'evasion_chance': 0.0,
            'detection_range': 500, 'aggressiveness': 0.0
        }

    # Core training methods
    def add_vessels_from_ai_scenario(self, ai_scenario):
        """Add vessels from AI scenario with better distribution"""
        vessels_added = 0
        max_vessels = min(8, len(ai_scenario))
        
        for vessel_data in ai_scenario[:max_vessels]:
            try:
                vessel = self.simulation.fleet.add_vessel(
                    x=vessel_data['x'], y=vessel_data['y'], vx=0.0, vy=0.0,
                    vessel_type=vessel_data['vessel_type'],
                    true_threat_level=vessel_data['true_threat_level'],
                    crew_count=vessel_data.get('crew_count', random.randint(2, 25))
                )
                if vessel not in self.simulation.units:
                    self.simulation.units.append(vessel)
                    vessels_added += 1
            except Exception as e:
                if self.enable_debug:
                    logger.warning(f"Error adding vessel: {e}")
                continue
        
        logger.info(f"Added {vessels_added} vessels to simulation")
        return vessels_added

    def execute_ai_decision(self, ai_report: AIReport, vessel: Vessel):
        """Execute AI decision and track outcomes"""
        try:
            action = ai_report.recommended_action
            true_threat = vessel.true_threat_level
            
            # Track threat distribution
            self.training_metrics['threat_type_breakdown'][true_threat] += 1
            
            outcome_map = {
                AIAction.INTERCEPT: self._handle_intercept_action(vessel, true_threat),
                AIAction.MONITOR: self._handle_monitor_action(vessel, true_threat),
                AIAction.IGNORE: self._handle_ignore_action(vessel, true_threat),
                AIAction.AWAIT_CONFIRMATION: self._handle_hitl_action(vessel, true_threat),
                AIAction.EVADE: "evasion"  # Default for evade action
            }
            
            return outcome_map.get(action, "no_action")
        except Exception as e:
            logger.warning(f"Error executing AI decision: {e}")
            return "error"

    def _handle_intercept_action(self, vessel: Vessel, true_threat: str):
        """Handle intercept action outcomes"""
        if true_threat in ["possible", "confirmed"]:
            vessel.active = False
            self.training_metrics['action_counts']['successful_intercepts'] += 1
            return "successful_intercept"
        else:
            self.training_metrics['action_counts']['false_positives'] += 1
            return "false_positive"

    def _handle_monitor_action(self, vessel: Vessel, true_threat: str):
        """Handle monitor action outcomes"""
        if true_threat == "possible":
            self.training_metrics['action_counts']['correct_monitors'] += 1
            return "correct_monitor"
        elif true_threat == "confirmed":
            self.training_metrics['action_counts']['missed_threats'] += 1
            return "missed_threat"
        return "monitoring"

    def _handle_ignore_action(self, vessel: Vessel, true_threat: str):
        """Handle ignore action outcomes"""
        if true_threat == "neutral":
            self.training_metrics['action_counts']['correct_ignores'] += 1
            return "correct_ignore"
        elif true_threat in ["possible", "confirmed"]:
            self.training_metrics['action_counts']['missed_threats'] += 1
            return "missed_threat"
        return "ignored"

    def _handle_hitl_action(self, vessel: Vessel, true_threat: str):
        """Handle HITL action outcomes"""
        self.training_metrics['action_counts']['hitl_requests'] += 1
        if true_threat in ["possible", "confirmed"]:
            vessel.active = False
            self.training_metrics['action_counts']['successful_intercepts'] += 1
            return "human_confirmed_threat"
        return "human_confirmed_safe"

    def run_training_episode(self, episode_num: int):
        """Run training episode with comprehensive tracking"""
        logger.info(f"Starting training episode {episode_num}")
        
        try:
            # Progressive difficulty
            difficulty = "easy" if episode_num < 10 else "medium" if episode_num < 30 else "hard"
            ai_scenario = self.ai.generate_realistic_scenario(difficulty)
            
            if self.add_vessels_from_ai_scenario(ai_scenario) == 0:
                logger.warning("No vessels added, skipping episode")
                self.training_metrics['episode_rewards'].append(0.0)
                self.training_metrics['episode_threat_accuracy'].append(0.0)
                return
            
            episode_reward, threat_counts = self._run_episode_steps(episode_num)
            episode_accuracy = self._calculate_episode_accuracy()
            
            self.training_metrics['episode_threat_accuracy'].append(episode_accuracy)
            self.training_metrics['episode_rewards'].append(episode_reward)
            
            logger.info(f"Episode {episode_num} completed. Reward: {episode_reward:.2f}, "
                       f"Accuracy: {episode_accuracy:.2%}, Threats: {threat_counts}")
            
        except Exception as e:
            logger.error(f"Error in training episode {episode_num}: {e}")
            self.training_metrics['episode_rewards'].append(0.0)
            self.training_metrics['episode_threat_accuracy'].append(0.0)
        finally:
            self._reset_episode()

    def _run_episode_steps(self, episode_num: int):
        """Run steps for a single episode"""
        episode_reward = 0
        player_vessel_ai = self._convert_player_to_ai_format()
        threat_counts = {"confirmed": 0, "possible": 0, "neutral": 0}
        
        for step in range(self.steps_per_episode):
            if self.simulation.game_over:
                break
                
            self.simulation.update_simulation()
            active_vessels = [v for v in self.simulation.units if v.active and v != self.simulation.player_ship]
            
            if not active_vessels:
                break
            
            for vessel in active_vessels:
                episode_reward += self._process_vessel_decision(vessel, player_vessel_ai, threat_counts)
        
        return episode_reward, threat_counts

    def _process_vessel_decision(self, vessel, player_vessel_ai, threat_counts):
        """Process AI decision for a single vessel"""
        try:
            threat_level = vessel.true_threat_level
            threat_counts[threat_level] += 1
            
            vessel_ai_format = self._convert_vessel_to_ai_format({
                'id': vessel.id, 'x': vessel.x, 'y': vessel.y,
                'speed': vessel.speed, 'heading': vessel.heading,
                'vessel_type': vessel.vessel_type, 'true_threat_level': threat_level
            })
            
            ai_report = self.ai.decide_action(vessel_ai_format, player_vessel_ai)
            reward = self.reward_calculator.calculate_reward(ai_report, vessel, "ai_decision")
            
            self._update_confidence_stats(ai_report.confidence)
            self.execute_ai_decision(ai_report, vessel)
            
            # Train AI
            current_state = self.ai.get_state(vessel_ai_format, player_vessel_ai)
            action_index = list(AIAction).index(ai_report.recommended_action)
            
            self.ai.record_and_train(
                state=current_state, action_index=action_index,
                reward=reward, next_state=current_state,
                done=False, human_feedback=None
            )
            
            return reward
        except Exception as e:
            if self.enable_debug:
                logger.warning(f"Error processing vessel {vessel.id}: {e}")
            return 0

    def _update_confidence_stats(self, confidence: float):
        """Update confidence statistics"""
        if confidence > 0.7:
            self.training_metrics['confidence_stats']['high'] += 1
        elif confidence > 0.4:
            self.training_metrics['confidence_stats']['medium'] += 1
        else:
            self.training_metrics['confidence_stats']['low'] += 1

    def _calculate_episode_accuracy(self):
        """Calculate threat detection accuracy for the episode"""
        action_counts = self.training_metrics['action_counts']
        total_actions = sum(action_counts.values())
        
        if total_actions == 0:
            return 0.0
            
        correct_actions = (action_counts['successful_intercepts'] +
                          action_counts['correct_monitors'] +
                          action_counts['correct_ignores'])
        
        return correct_actions / total_actions

    def _reset_episode(self):
        """Reset simulation for next episode"""
        try:
            # Remove all non-player vessels
            vessels_to_remove = [v for v in self.simulation.units if v != self.simulation.player_ship]
            for vessel in vessels_to_remove:
                vessel.active = False
                if hasattr(vessel, 'id') and vessel.id in self.simulation.fleet.vessels:
                    del self.simulation.fleet.vessels[vessel.id]
            
            self.simulation.units = [self.simulation.player_ship]
            self.simulation.player_ship.x = 100.0
            self.simulation.player_ship.y = 100.0
            self.simulation.player_ship.set_velocity(0, 0)
            
        except Exception as e:
            logger.warning(f"Error resetting episode: {e}")

    def train(self):
        """Enhanced training loop with comprehensive monitoring"""
        print("\n🚀 ENHANCED AI TRAINING PIPELINE")
        print("=" * 50)
        
        if not self.run_comprehensive_test():
            print("❌ System tests failed. Cannot start training.")
            return
        
        print("\n🎯 STARTING TRAINING")
        print("=" * 50)
        
        start_time = time.time()
        
        for episode in range(self.episodes):
            try:
                self.run_training_episode(episode)
                
                # Progress tracking
                if episode % 5 == 0 or episode == self.episodes - 1:
                    recent_rewards = self.training_metrics['episode_rewards'][-5:]
                    recent_accuracy = self.training_metrics['episode_threat_accuracy'][-5:]
                    
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                    avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.0
                    
                    print(f"📊 Episode {episode:3d}: "
                          f"Avg Reward: {avg_reward:7.2f} | "
                          f"Accuracy: {avg_accuracy:6.1%} | "
                          f"Epsilon: {self.ai.epsilon:.3f}")
                
            except Exception as e:
                logger.error(f"Critical error in episode {episode}: {e}")
                continue
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        print(f"📈 Successful episodes: {self.episodes}/{self.episodes}")
        
        self._print_comprehensive_report()

    def _print_comprehensive_report(self):
        """Generate comprehensive training report"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED AI TRAINING - COMPREHENSIVE REPORT")
        print("=" * 60)
        
        try:
            # Basic metrics
            if self.training_metrics['episode_rewards']:
                avg_reward = np.mean(self.training_metrics['episode_rewards'])
                std_reward = np.std(self.training_metrics['episode_rewards'])
                avg_accuracy = np.mean(self.training_metrics['episode_threat_accuracy'])
            else:
                avg_reward = std_reward = avg_accuracy = 0.0
            
            action_counts = self.training_metrics['action_counts']
            total_actions = sum(action_counts.values())
            
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  Episodes Completed: {len(self.training_metrics['episode_rewards'])}")
            print(f"  Average Reward: {avg_reward:7.2f} ± {std_reward:6.2f}")
            print(f"  Threat Detection Accuracy: {avg_accuracy:7.1%}")
            
            print(f"\n🎯 ACTION BREAKDOWN:")
            for action, count in action_counts.items():
                print(f"  {action.replace('_', ' ').title():20} {count:4d}")
            
            # AI performance
            ai_perf = self._get_ai_performance_report()
            print(f"\n🧠 AI ENGINE PERFORMANCE:")
            print(f"  Total Decisions:   {ai_perf.get('total_decisions', 0):6d}")
            print(f"  Autonomous Rate:   {ai_perf.get('autonomous_success_rate', '0%'):>9}")
            print(f"  Training Steps:    {ai_perf.get('training_steps', 0):6d}")
            print(f"  Final Epsilon:     {ai_perf.get('current_epsilon', 0.0):9.4f}")
            
        except Exception as e:
            print(f"Error generating report: {e}")
        
        print("=" * 60)

    def _get_ai_performance_report(self):
        """Safely get AI performance report with error handling"""
        try:
            return self.ai.get_performance_report()
        except Exception as e:
            logger.warning(f"Error getting AI performance report: {e}")
            return {
                "total_decisions": getattr(self.ai, 'performance_metrics', {}).get('decisions', 0),
                "autonomous_success_rate": "0%",
                "training_steps": getattr(self.ai, 'train_counter', 0),
                "current_epsilon": getattr(self.ai, 'epsilon', 0.0)
            }

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Naval Combat AI Training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=40, help='Steps per episode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test-only', action='store_true', help='Run tests only')
    
    args = parser.parse_args()
    
    if args.test_only:
        pipeline = EnhancedAITrainingPipeline(enable_debug=True)
        pipeline.run_comprehensive_test()
    else:
        pipeline = EnhancedAITrainingPipeline(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            enable_debug=args.debug
        )
        pipeline.train()

if __name__ == "__main__":
    main()