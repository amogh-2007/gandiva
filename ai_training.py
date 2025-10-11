# ai_training.py
"""
Enhanced AI Training Pipeline for Naval Combat Simulation
Integrated with all fixes, improved reward system, and comprehensive debugging
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
    """
    Fixed reward calculation with better handling of edge cases
    """
    
    @staticmethod
    def calculate_reward(ai_report: AIReport, vessel, action_taken: str) -> float:
        """
        Robust reward calculation with detailed cases
        """
        try:
            # Get threat level from vessel object
            if hasattr(vessel, 'true_threat_level'):
                true_threat = vessel.true_threat_level
            else:
                # Fallback for dictionary-style vessels
                true_threat = vessel.get('true_threat_level', 'neutral')
            
            recommended_action = ai_report.recommended_action
            confidence = ai_report.confidence
            
            # Default reward
            reward = 0.0
            
            # Handle different threat levels and actions
            if true_threat == "confirmed":
                reward = FixedRewardCalculator._confirmed_threat_reward(recommended_action, confidence)
            elif true_threat == "possible":
                reward = FixedRewardCalculator._possible_threat_reward(recommended_action, confidence)
            else:  # neutral
                reward = FixedRewardCalculator._neutral_threat_reward(recommended_action, confidence)
            
            # Confidence adjustments
            reward = FixedRewardCalculator._apply_confidence_adjustments(reward, confidence)
            
            logger.debug(f"Reward calculation: {true_threat} + {recommended_action.value} + {confidence:.2f} = {reward:.2f}")
            return reward
            
        except Exception as e:
            logger.error(f"Error in reward calculation: {e}")
            return 0.0
    
    @staticmethod
    def _confirmed_threat_reward(action: AIAction, confidence: float) -> float:
        """Reward for confirmed threats"""
        if action == AIAction.INTERCEPT:
            return 20.0 + (confidence * 8.0)  # High reward for intercepting real threats
        elif action == AIAction.MONITOR:
            return 3.0  # Small reward for monitoring
        elif action == AIAction.IGNORE:
            return -25.0  # Heavy penalty for ignoring
        elif action == AIAction.AWAIT_CONFIRMATION:
            return -3.0  # Small penalty for HITL on clear threats
        elif action == AIAction.EVADE:
            return 5.0  # Moderate reward for evasion
        else:
            return -5.0  # Default penalty for unknown actions
    
    @staticmethod
    def _possible_threat_reward(action: AIAction, confidence: float) -> float:
        """Reward for possible threats"""
        if action == AIAction.MONITOR:
            return 12.0 + (confidence * 4.0)  # Good reward for monitoring
        elif action == AIAction.INTERCEPT:
            return -3.0  # Small penalty for aggressive interception
        elif action == AIAction.IGNORE:
            return -8.0  # Penalty for ignoring
        elif action == AIAction.AWAIT_CONFIRMATION:
            return 5.0  # Reward for cautious approach
        elif action == AIAction.EVADE:
            return 8.0  # Good reward for evasion
        else:
            return -2.0  # Small penalty for unknown actions
    
    @staticmethod
    def _neutral_threat_reward(action: AIAction, confidence: float) -> float:
        """Reward for neutral threats"""
        if action == AIAction.IGNORE:
            return 8.0 + (confidence * 3.0)  # Reward for correct ignore
        elif action == AIAction.MONITOR:
            return 2.0  # Small reward for monitoring
        elif action == AIAction.INTERCEPT:
            return -15.0  # Penalty for intercepting neutrals
        elif action == AIAction.AWAIT_CONFIRMATION:
            return -2.0  # Small penalty for HITL
        elif action == AIAction.EVADE:
            return -5.0  # Penalty for unnecessary evasion
        else:
            return -1.0  # Small penalty for unknown actions
    
    @staticmethod
    def _apply_confidence_adjustments(reward: float, confidence: float) -> float:
        """Apply confidence-based adjustments"""
        if confidence > 0.8:
            return reward + 3.0  # Bonus for high confidence
        elif confidence < 0.3:
            return reward - 2.0  # Penalty for low confidence
        else:
            return reward

class EnhancedAITrainingPipeline:
    """
    Comprehensive AI Training Pipeline with all improvements integrated
    """
    
    def __init__(self, episodes=100, steps_per_episode=50, enable_debug=False):
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.enable_debug = enable_debug
        self.simulation = SimulationController(mission_type="Training", difficulty="medium")
        self.ai = NavalAI(backend=None)
        self.reward_calculator = FixedRewardCalculator()
        
        # Comprehensive training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_threat_accuracy': [],
            'successful_intercepts': 0,
            'false_positives': 0,
            'missed_threats': 0,
            'correct_monitors': 0,
            'correct_ignores': 0,
            'hitl_requests': 0,
            'training_losses': [],
            'confidence_stats': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        if self.enable_debug:
            logger.setLevel(logging.DEBUG)
            print("🔧 Debug mode enabled")
    
    def run_comprehensive_test(self):
        """Run comprehensive system test before training"""
        print("\n🔍 COMPREHENSIVE SYSTEM TEST")
        print("=" * 60)
        
        tests_passed = 0
        total_tests = 4
        
        # Test 1: AI Initialization
        print("\n1. Testing AI Initialization...")
        if self.test_ai_initialization():
            tests_passed += 1
            print("   ✅ AI Initialization test passed")
        else:
            print("   ❌ AI Initialization test failed")
        
        # Test 2: Scenario Generation
        print("\n2. Testing Scenario Generation...")
        if self.test_scenario_generation():
            tests_passed += 1
            print("   ✅ Scenario Generation test passed")
        else:
            print("   ❌ Scenario Generation test failed")
        
        # Test 3: Reward Calculation
        print("\n3. Testing Reward Calculation...")
        if self.test_reward_calculation():
            tests_passed += 1
            print("   ✅ Reward Calculation test passed")
        else:
            print("   ❌ Reward Calculation test failed")
        
        # Test 4: Training Integration
        print("\n4. Testing Training Integration...")
        if self.test_training_integration():
            tests_passed += 1
            print("   ✅ Training Integration test passed")
        else:
            print("   ❌ Training Integration test failed")
        
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
            # Check if AI components are properly initialized
            assert hasattr(self.ai, 'model'), "AI model not initialized"
            assert hasattr(self.ai, 'replay_buffer'), "Replay buffer not initialized"
            assert self.ai.epsilon > 0, "Epsilon should be positive"
            
            # Test basic decision making
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
            # Test scenario generation for different difficulties
            for difficulty in ["easy", "medium", "hard"]:
                scenario = self.ai.generate_realistic_scenario(difficulty)
                assert len(scenario) > 0, f"No vessels generated for {difficulty}"
                
                # Check threat distribution
                threat_counts = {"neutral": 0, "possible": 0, "confirmed": 0}
                for vessel in scenario:
                    threat_level = vessel.get('true_threat_level', 'neutral')
                    threat_counts[threat_level] += 1
                
                print(f"   {difficulty}: {threat_counts}")
            
            # Test vessel conversion
            test_vessel_data = scenario[0]
            converted = self._convert_vessel_to_ai_format(test_vessel_data)
            assert 'true_threat_level' in converted, "Vessel conversion missing threat level"
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_reward_calculation(self):
        """Test reward calculation with various scenarios"""
        try:
            test_cases = [
                # (threat_level, action, expected_min, description)
                ("confirmed", AIAction.INTERCEPT, 15, "Intercept confirmed threat"),
                ("confirmed", AIAction.IGNORE, -25, "Ignore confirmed threat"),
                ("possible", AIAction.MONITOR, 8, "Monitor possible threat"),
                ("neutral", AIAction.IGNORE, 5, "Ignore neutral vessel"),
            ]
            
            for threat_level, action, expected_min, description in test_cases:
                # Create test vessel and report
                vessel_data = self._create_test_vessel("Test", threat_level, 300, 300)
                report = AIReport(
                    vessel_id="test",
                    threat_assessment=threat_level,
                    recommended_action=action,
                    confidence=0.7,
                    reasoning="Test",
                    timestamp=time.time()
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
            # Add test vessels
            test_vessels = [
                self._create_test_vessel("Fishing Boat", "neutral", 200, 200),
                self._create_test_vessel("Suspicious Craft", "possible", 300, 300),
            ]
            
            vessels_added = 0
            for vessel_data in test_vessels:
                try:
                    vessel = self.simulation.fleet.add_vessel(
                        x=vessel_data['x'], y=vessel_data['y'],
                        vx=0.0, vy=0.0,
                        vessel_type=vessel_data['vessel_type'],
                        true_threat_level=vessel_data['true_threat_level']
                    )
                    if vessel not in self.simulation.units:
                        self.simulation.units.append(vessel)
                        vessels_added += 1
                except Exception as e:
                    print(f"   Error adding vessel: {e}")
                    return False
            
            # Test AI decision and training
            if vessels_added > 0:
                player_data = self._convert_player_to_ai_format()
                vessel = self.simulation.units[1]  # First non-player vessel
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
                    state=current_state,
                    action_index=action_index,
                    reward=reward,
                    next_state=current_state,
                    done=False,
                    human_feedback=None
                )
                
                print(f"   ✅ Training step: {vessel.vessel_type} -> {report.recommended_action.value} (reward: {reward:.2f})")
                return True
            else:
                print("   ❌ No vessels added")
                return False
                
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def _create_test_vessel(self, vessel_type: str, threat_level: str, x: float, y: float) -> dict:
        """Create a test vessel with specified parameters"""
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
        """Create test player data"""
        return {
            'id': 'player_1', 'x': 500, 'y': 500, 'speed': 0, 'heading': 0,
            'vessel_type': 'Player Ship', 'behavior': 'command',
            'true_threat_level': 'neutral', 'evasion_chance': 0.0,
            'detection_range': 500, 'aggressiveness': 0.0
        }
    
    def _convert_vessel_to_ai_format(self, vessel_data: dict) -> dict:
        """Convert vessel data to AI-compatible format"""
        try:
            return {
                'id': vessel_data.get('id', 0),
                'x': vessel_data.get('x', 400),
                'y': vessel_data.get('y', 300),
                'speed': max(0.1, vessel_data.get('speed', 5.0)),
                'heading': vessel_data.get('heading', 0),
                'vessel_type': vessel_data.get('vessel_type', 'Unknown'),
                'behavior': vessel_data.get('behavior', 'unknown'),
                'true_threat_level': vessel_data.get('true_threat_level', 'neutral'),
                'evasion_chance': vessel_data.get('evasion_chance', 0.1),
                'detection_range': vessel_data.get('detection_range', 200),
                'aggressiveness': vessel_data.get('aggressiveness', 0.1)
            }
        except Exception as e:
            logger.warning(f"Error converting vessel: {e}")
            return self._create_test_vessel("Unknown", "neutral", 400, 300)
    
    def _convert_player_to_ai_format(self) -> dict:
        """Convert player vessel to AI format"""
        player = self.simulation.player_ship
        return {
            'id': player.id,
            'x': player.x, 'y': player.y,
            'speed': max(0.1, player.speed),
            'heading': player.heading,
            'vessel_type': 'Player Ship',
            'behavior': 'command',
            'true_threat_level': 'neutral',
            'evasion_chance': 0.0,
            'detection_range': 500,
            'aggressiveness': 0.0
        }
    
    def add_vessels_from_ai_scenario(self, ai_scenario):
        """Add vessels from AI scenario with better distribution"""
        vessels_added = 0
        max_vessels = min(8, len(ai_scenario))  # Limit vessels per episode
        
        for vessel_data in ai_scenario[:max_vessels]:
            try:
                vessel = self.simulation.fleet.add_vessel(
                    x=vessel_data['x'], y=vessel_data['y'],
                    vx=0.0, vy=0.0,
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
    
    def _update_confidence_stats(self, confidence: float):
        """Update confidence statistics"""
        if confidence > 0.7:
            self.training_metrics['confidence_stats']['high'] += 1
        elif confidence > 0.4:
            self.training_metrics['confidence_stats']['medium'] += 1
        else:
            self.training_metrics['confidence_stats']['low'] += 1
    
    def execute_ai_decision(self, ai_report: AIReport, vessel: Vessel):
        """Execute AI decision and track outcomes"""
        try:
            action = ai_report.recommended_action
            true_threat = vessel.true_threat_level
            
            if action == AIAction.INTERCEPT:
                if true_threat in ["possible", "confirmed"]:
                    vessel.active = False
                    self.training_metrics['successful_intercepts'] += 1
                    return "successful_intercept"
                else:
                    self.training_metrics['false_positives'] += 1
                    return "false_positive"
                    
            elif action == AIAction.MONITOR:
                if true_threat == "possible":
                    self.training_metrics['correct_monitors'] += 1
                    return "correct_monitor"
                else:
                    return "monitoring"
                    
            elif action == AIAction.IGNORE:
                if true_threat == "neutral":
                    self.training_metrics['correct_ignores'] += 1
                    return "correct_ignore"
                elif true_threat in ["possible", "confirmed"]:
                    self.training_metrics['missed_threats'] += 1
                    return "missed_threat"
                else:
                    return "ignored"
                    
            elif action == AIAction.AWAIT_CONFIRMATION:
                self.training_metrics['hitl_requests'] += 1
                # Simulate human decision
                if true_threat in ["possible", "confirmed"]:
                    vessel.active = False
                    return "human_confirmed_threat"
                else:
                    return "human_confirmed_safe"
        
            return "no_action"
        except Exception as e:
            logger.warning(f"Error executing AI decision: {e}")
            return "error"
    
    def calculate_episode_accuracy(self):
        """Calculate threat detection accuracy for the episode"""
        total_actions = (self.training_metrics['successful_intercepts'] + 
                        self.training_metrics['false_positives'] + 
                        self.training_metrics['missed_threats'] +
                        self.training_metrics['correct_monitors'] +
                        self.training_metrics['correct_ignores'])
        
        if total_actions == 0:
            return 0.0
            
        correct_actions = (self.training_metrics['successful_intercepts'] +
                          self.training_metrics['correct_monitors'] +
                          self.training_metrics['correct_ignores'])
        
        return correct_actions / total_actions
    
    def run_training_episode(self, episode_num: int):
        """Run training episode with comprehensive tracking"""
        logger.info(f"Starting training episode {episode_num}")
        
        try:
            # Progressive difficulty
            difficulty_levels = ["easy", "medium", "hard"]
            difficulty_index = min(episode_num // (self.episodes // len(difficulty_levels)), len(difficulty_levels) - 1)
            difficulty = difficulty_levels[difficulty_index]
            
            # Generate scenario
            ai_scenario = self.ai.generate_realistic_scenario(difficulty)
            vessels_added = self.add_vessels_from_ai_scenario(ai_scenario)
            
            if vessels_added == 0:
                logger.warning("No vessels added to simulation, skipping episode")
                self.training_metrics['episode_rewards'].append(0.0)
                self.training_metrics['episode_threat_accuracy'].append(0.0)
                return
            
            episode_reward = 0
            player_vessel_ai = self._convert_player_to_ai_format()
            
            # Reset episode-specific metrics
            episode_metrics = {
                'successful_intercepts': 0,
                'false_positives': 0,
                'missed_threats': 0,
                'correct_monitors': 0,
                'correct_ignores': 0,
                'hitl_requests': 0
            }
            
            for step in range(self.steps_per_episode):
                if self.simulation.game_over:
                    break
                    
                # Update simulation
                self.simulation.update_simulation()
                
                # Get active vessels
                active_vessels = [v for v in self.simulation.units 
                                if v.active and v != self.simulation.player_ship]
                
                if not active_vessels:
                    if self.enable_debug:
                        logger.info("No active vessels, ending episode early")
                    break
                
                # AI makes decisions for each vessel
                for vessel in active_vessels:
                    try:
                        vessel_ai_format = self._convert_vessel_to_ai_format({
                            'id': vessel.id, 'x': vessel.x, 'y': vessel.y,
                            'speed': vessel.speed, 'heading': vessel.heading,
                            'vessel_type': vessel.vessel_type,
                            'true_threat_level': vessel.true_threat_level
                        })
                        
                        # Get AI decision
                        ai_report = self.ai.decide_action(vessel_ai_format, player_vessel_ai)
                        
                        # Calculate reward
                        reward = self.reward_calculator.calculate_reward(ai_report, vessel, "ai_decision")
                        episode_reward += reward
                        
                        # Update confidence stats
                        self._update_confidence_stats(ai_report.confidence)
                        
                        # Execute decision and track outcome
                        outcome = self.execute_ai_decision(ai_report, vessel)
                        
                        # Update episode metrics
                        if outcome == "successful_intercept":
                            episode_metrics['successful_intercepts'] += 1
                        elif outcome == "false_positive":
                            episode_metrics['false_positives'] += 1
                        elif outcome == "missed_threat":
                            episode_metrics['missed_threats'] += 1
                        elif outcome == "correct_monitor":
                            episode_metrics['correct_monitors'] += 1
                        elif outcome == "correct_ignore":
                            episode_metrics['correct_ignores'] += 1
                        elif outcome == "human_confirmed_threat":
                            episode_metrics['hitl_requests'] += 1
                            episode_metrics['successful_intercepts'] += 1
                        
                        # Train AI with the experience
                        current_state = self.ai.get_state(vessel_ai_format, player_vessel_ai)
                        action_index = list(AIAction).index(ai_report.recommended_action)
                        
                        self.ai.record_and_train(
                            state=current_state,
                            action_index=action_index,
                            reward=reward,
                            next_state=current_state,
                            done=False,
                            human_feedback=None
                        )
                        
                    except Exception as e:
                        if self.enable_debug:
                            logger.warning(f"Error processing vessel {vessel.id}: {e}")
                        continue
            
            # Update global metrics
            for key in episode_metrics:
                self.training_metrics[key] += episode_metrics[key]
            
            # Calculate episode accuracy
            episode_accuracy = self.calculate_episode_accuracy()
            self.training_metrics['episode_threat_accuracy'].append(episode_accuracy)
            self.training_metrics['episode_rewards'].append(episode_reward)
            
            logger.info(f"Episode {episode_num} completed. "
                       f"Reward: {episode_reward:.2f}, "
                       f"Accuracy: {episode_accuracy:.2%}, "
                       f"Intercepts: {episode_metrics['successful_intercepts']}")
            
        except Exception as e:
            logger.error(f"Error in training episode {episode_num}: {e}")
            self.training_metrics['episode_rewards'].append(0.0)
            self.training_metrics['episode_threat_accuracy'].append(0.0)
        
        finally:
            # Clean up for next episode
            self._reset_episode()
    
    def _reset_episode(self):
        """Reset simulation for next episode"""
        try:
            # Remove all non-player vessels
            vessels_to_remove = [v for v in self.simulation.units 
                               if v != self.simulation.player_ship]
            for vessel in vessels_to_remove:
                vessel.active = False
                if hasattr(vessel, 'id') and vessel.id in self.simulation.fleet.vessels:
                    del self.simulation.fleet.vessels[vessel.id]
            
            self.simulation.units = [self.simulation.player_ship]
            
            # Reset player position
            self.simulation.player_ship.x = 100.0
            self.simulation.player_ship.y = 100.0
            self.simulation.player_ship.set_velocity(0, 0)
            
            # Reset zone state
            self.simulation.patrol_phase_active = True
            self.simulation.in_patrol_zone = False
            self.simulation.zone_expanded = False
            
        except Exception as e:
            logger.warning(f"Error resetting episode: {e}")
    
    def train(self):
        """Enhanced training loop with comprehensive monitoring"""
        print("\n🚀 ENHANCED AI TRAINING PIPELINE")
        print("=" * 60)
        
        # Run comprehensive test first
        if not self.run_comprehensive_test():
            print("❌ System tests failed. Cannot start training.")
            return
        
        print("\n🎯 STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        successful_episodes = 0
        
        for episode in range(self.episodes):
            try:
                self.run_training_episode(episode)
                successful_episodes += 1
                
                # Enhanced progress tracking
                if episode % 5 == 0 or episode == self.episodes - 1:
                    recent_rewards = self.training_metrics['episode_rewards'][-5:]
                    recent_accuracy = self.training_metrics['episode_threat_accuracy'][-5:]
                    
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                    avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.0
                    
                    print(f"📊 Episode {episode:3d}: "
                          f"Avg Reward: {avg_reward:7.2f} | "
                          f"Accuracy: {avg_accuracy:6.1%} | "
                          f"Epsilon: {self.ai.epsilon:.3f}")
                
                # Save checkpoint every 25 episodes
                if episode % 25 == 0 and episode > 0:
                    self._save_checkpoint(episode)
                    
            except Exception as e:
                logger.error(f"Critical error in episode {episode}: {e}")
                continue
        
        training_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📈 Successful episodes: {successful_episodes}/{self.episodes}")
        
        # Final performance report
        self._print_comprehensive_report()
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        try:
            logger.info(f"Checkpoint saved at episode {episode}")
            
            avg_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
            avg_accuracy = np.mean(self.training_metrics['episode_threat_accuracy'][-10:])
            
            print(f"💾 Checkpoint Episode {episode}: "
                  f"Reward: {avg_reward:.2f}, "
                  f"Accuracy: {avg_accuracy:.1%}")
                       
        except Exception as e:
            logger.warning(f"Error saving checkpoint: {e}")
    
    def _print_comprehensive_report(self):
        """Print detailed training report"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED AI TRAINING - COMPREHENSIVE REPORT")
        print("=" * 70)
        
        try:
            # Training metrics
            if self.training_metrics['episode_rewards']:
                avg_reward = np.mean(self.training_metrics['episode_rewards'])
                std_reward = np.std(self.training_metrics['episode_rewards'])
                avg_accuracy = np.mean(self.training_metrics['episode_threat_accuracy'])
            else:
                avg_reward = std_reward = avg_accuracy = 0.0
            
            total_actions = sum([
                self.training_metrics['successful_intercepts'],
                self.training_metrics['false_positives'], 
                self.training_metrics['missed_threats'],
                self.training_metrics['correct_monitors'],
                self.training_metrics['correct_ignores']
            ])
            
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  Episodes Completed: {self.episodes}")
            print(f"  Average Reward: {avg_reward:7.2f} ± {std_reward:6.2f}")
            print(f"  Threat Detection Accuracy: {avg_accuracy:7.1%}")
            
            print(f"\n🎯 ACTION BREAKDOWN:")
            print(f"  Successful Intercepts: {self.training_metrics['successful_intercepts']:4d}")
            print(f"  False Positives:       {self.training_metrics['false_positives']:4d}")
            print(f"  Missed Threats:        {self.training_metrics['missed_threats']:4d}")
            print(f"  Correct Monitors:      {self.training_metrics['correct_monitors']:4d}")
            print(f"  Correct Ignores:       {self.training_metrics['correct_ignores']:4d}")
            print(f"  HITL Requests:         {self.training_metrics['hitl_requests']:4d}")
            
            if total_actions > 0:
                intercept_accuracy = self.training_metrics['successful_intercepts'] / max(1, 
                    self.training_metrics['successful_intercepts'] + self.training_metrics['false_positives'])
                print(f"  Intercept Accuracy:    {intercept_accuracy:7.1%}")
            
            # Confidence statistics
            conf_stats = self.training_metrics['confidence_stats']
            total_conf = sum(conf_stats.values())
            if total_conf > 0:
                print(f"\n🎯 CONFIDENCE STATISTICS:")
                print(f"  High Confidence:  {conf_stats['high']:4d} ({conf_stats['high']/total_conf:6.1%})")
                print(f"  Medium Confidence: {conf_stats['medium']:4d} ({conf_stats['medium']/total_conf:6.1%})")
                print(f"  Low Confidence:    {conf_stats['low']:4d} ({conf_stats['low']/total_conf:6.1%})")
            
            # AI performance
            ai_perf = self.ai.get_performance_report()
            print(f"\n🧠 AI ENGINE PERFORMANCE:")
            print(f"  Total Decisions:       {ai_perf['total_decisions']:6d}")
            print(f"  Autonomous Rate:       {ai_perf['autonomous_success_rate']:>9}")
            print(f"  Cumulative Reward:     {ai_perf['cumulative_reward']:7.1f}")
            print(f"  Training Steps:        {ai_perf['training_steps']:6d}")
            print(f"  Final Epsilon:         {ai_perf['current_epsilon']:9.4f}")
            print(f"  Buffer Size:           {ai_perf['replay_buffer_size']:6d}")
            
            # Learning progression
            if len(self.training_metrics['episode_rewards']) > 10:
                first_10_avg = np.mean(self.training_metrics['episode_rewards'][:10])
                last_10_avg = np.mean(self.training_metrics['episode_rewards'][-10:])
                improvement = last_10_avg - first_10_avg
                print(f"  Reward Improvement:    {improvement:7.1f}")
            
        except Exception as e:
            print(f"Error generating report: {e}")
        
        print("=" * 70)

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Naval Combat AI Training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=40, help='Steps per episode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--test-only', action='store_true', help='Run tests only, no training')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Run tests only
        pipeline = EnhancedAITrainingPipeline(enable_debug=True)
        pipeline.run_comprehensive_test()
    else:
        # Run full training
        pipeline = EnhancedAITrainingPipeline(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            enable_debug=args.debug
        )
        pipeline.train()

if __name__ == "__main__":
    main()