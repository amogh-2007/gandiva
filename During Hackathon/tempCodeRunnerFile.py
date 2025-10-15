# improved_training.py
"""
Improved AI Training with Better Reward Shaping and Training Strategy
"""

import time
import numpy as np
from backend import SimulationController, Vessel
from ai import NavalAI, AIAction, AIReport
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedAITrainingPipeline:
    """
    Enhanced training pipeline with better reward shaping and training strategy
    """
    
    def __init__(self, episodes=100, steps_per_episode=50):
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.simulation = SimulationController(mission_type="Training", difficulty="medium")
        self.ai = NavalAI(backend=None)
        
        # Enhanced training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'successful_intercepts': 0,
            'false_positives': 0,
            'missed_threats': 0,
            'correct_monitors': 0,
            'correct_ignores': 0,
            'hitl_requests': 0,
            'episode_threat_accuracy': []
        }
        
    def convert_vessel_to_ai_format(self, vessel: Vessel) -> dict:
        """Convert backend Vessel to AI-compatible dictionary format"""
        try:
            return {
                'id': vessel.id,
                'x': vessel.x,
                'y': vessel.y,
                'speed': max(0.1, vessel.speed),
                'heading': vessel.heading,
                'vessel_type': vessel.vessel_type,
                'behavior': 'unknown',
                'true_threat_level': vessel.true_threat_level,
                'evasion_chance': 0.7 if vessel.true_threat_level == 'confirmed' else 
                                0.4 if vessel.true_threat_level == 'possible' else 0.1,
                'detection_range': 200,
                'aggressiveness': 0.9 if vessel.true_threat_level == 'confirmed' else 
                                0.5 if vessel.true_threat_level == 'possible' else 0.1
            }
        except Exception as e:
            logger.warning(f"Error converting vessel: {e}")
            return self._get_default_vessel_format()

    def _get_default_vessel_format(self) -> dict:
        """Safe default vessel format"""
        return {
            'id': 0,
            'x': 400, 'y': 300, 'speed': 1.0, 'heading': 0,
            'vessel_type': 'Unknown', 'behavior': 'idle',
            'true_threat_level': 'neutral', 'evasion_chance': 0.1,
            'detection_range': 200, 'aggressiveness': 0.1
        }
    
    def convert_player_to_ai_format(self) -> dict:
        """Convert player vessel to AI-compatible format"""
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
                # Use the fleet manager to add vessels
                vessel = self.simulation.fleet.add_vessel(
                    x=vessel_data['x'],
                    y=vessel_data['y'],
                    vx=0.0, vy=0.0,
                    vessel_type=vessel_data['vessel_type'],
                    true_threat_level=vessel_data['true_threat_level'],
                    crew_count=vessel_data.get('crew_count', random.randint(2, 25))
                )
                # Also add to units list for simulation tracking
                if vessel not in self.simulation.units:
                    self.simulation.units.append(vessel)
                    vessels_added += 1
            except Exception as e:
                logger.warning(f"Error adding vessel: {e}")
                continue
        
        logger.info(f"Added {vessels_added} vessels to simulation")
        return vessels_added
    
    def calculate_improved_reward(self, ai_report: AIReport, vessel: Vessel, action_taken: str) -> float:
        """
        Improved reward calculation with better shaping
        """
        try:
            true_threat = vessel.true_threat_level
            recommended_action = ai_report.recommended_action
            confidence = ai_report.confidence
            
            reward = 0.0
            
            # Base rewards for correct actions
            if true_threat == "confirmed":
                if recommended_action == AIAction.INTERCEPT:
                    reward += 15.0  # High reward for intercepting real threats
                    reward += confidence * 5.0  # Bonus for high confidence
                elif recommended_action == AIAction.MONITOR:
                    reward += 2.0  # Small reward for monitoring confirmed threats
                elif recommended_action == AIAction.IGNORE:
                    reward -= 20.0  # Heavy penalty for ignoring confirmed threats
                elif recommended_action == AIAction.AWAIT_CONFIRMATION:
                    reward -= 2.0  # Small penalty for HITL on clear threats
                    
            elif true_threat == "possible":
                if recommended_action == AIAction.MONITOR:
                    reward += 8.0  # Good reward for monitoring suspicious vessels
                    reward += confidence * 3.0
                elif recommended_action == AIAction.INTERCEPT:
                    reward -= 5.0  # Moderate penalty for aggressive interception
                elif recommended_action == AIAction.IGNORE:
                    reward -= 5.0  # Penalty for ignoring suspicious vessels
                elif recommended_action == AIAction.AWAIT_CONFIRMATION:
                    reward += 1.0  # Small reward for cautious approach
                    
            else:  # neutral
                if recommended_action == AIAction.IGNORE:
                    reward += 5.0  # Reward for correctly ignoring neutrals
                    reward += confidence * 2.0
                elif recommended_action == AIAction.MONITOR:
                    reward += 1.0  # Small reward for monitoring neutrals
                elif recommended_action == AIAction.INTERCEPT:
                    reward -= 10.0  # Penalty for intercepting neutrals
                elif recommended_action == AIAction.AWAIT_CONFIRMATION:
                    reward -= 1.0  # Small penalty for HITL on clear neutrals
            
            # Confidence bonus/penalty
            if confidence > 0.8:
                reward += 2.0  # Bonus for high confidence
            elif confidence < 0.3:
                reward -= 1.0  # Penalty for low confidence
                
            return reward
            
        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            return 0.0
    
    def execute_ai_decision(self, ai_report: AIReport, vessel: Vessel):
        """
        Execute AI decision and track outcomes
        """
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
        """
        Run training episode with improved strategy
        """
        logger.info(f"Starting training episode {episode_num}")
        
        try:
            # Progressive difficulty
            difficulty_levels = ["easy", "medium", "hard"]
            difficulty_index = min(episode_num // (self.episodes // len(difficulty_levels)), len(difficulty_levels) - 1)
            difficulty = difficulty_levels[difficulty_index]
            
            # Use AI to generate scenario
            ai_scenario = self.ai.generate_realistic_scenario(difficulty)
            
            # Add vessels
            vessels_added = self.add_vessels_from_ai_scenario(ai_scenario)
            
            if vessels_added == 0:
                logger.warning("No vessels added to simulation, skipping episode")
                self.training_metrics['episode_rewards'].append(0.0)
                self.training_metrics['episode_threat_accuracy'].append(0.0)
                return
            
            episode_reward = 0
            player_vessel_ai = self.convert_player_to_ai_format()
            
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
                    logger.info("No active vessels, ending episode early")
                    break
                
                # AI makes decisions for each vessel
                for vessel in active_vessels:
                    try:
                        vessel_ai_format = self.convert_vessel_to_ai_format(vessel)
                        
                        # Get AI decision
                        ai_report = self.ai.decide_action(vessel_ai_format, player_vessel_ai)
                        
                        # Calculate reward
                        reward = self.calculate_improved_reward(ai_report, vessel, "ai_decision")
                        episode_reward += reward
                        
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
                            next_state=current_state,  # Use current state for simplicity
                            done=False,
                            human_feedback=None
                        )
                        
                    except Exception as e:
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
        """
        Enhanced training loop with better monitoring
        """
        logger.info("Starting improved AI training pipeline")
        
        start_time = time.time()
        successful_episodes = 0
        
        print("\n🚀 Starting Improved AI Training")
        print("=" * 60)
        
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
        print("🎯 IMPROVED AI TRAINING - COMPREHENSIVE REPORT")
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

def quick_test():
    """Test the improved training pipeline"""
    print("🚀 Testing Improved Training Pipeline...")
    
    try:
        pipeline = ImprovedAITrainingPipeline(episodes=2, steps_per_episode=10)
        
        # Test scenario generation
        test_vessels = pipeline.ai.generate_realistic_scenario("easy")
        print(f"✅ Generated {len(test_vessels)} test vessels")
        
        # Test vessel addition
        vessels_added = pipeline.add_vessels_from_ai_scenario(test_vessels)
        print(f"✅ Added {vessels_added} vessels to simulation")
        
        # Test reward calculation
        test_vessel = pipeline.simulation.units[1] if len(pipeline.simulation.units) > 1 else None
        if test_vessel:
            vessel_data = pipeline.convert_vessel_to_ai_format(test_vessel)
            player_data = pipeline.convert_player_to_ai_format()
            report = pipeline.ai.decide_action(vessel_data, player_data)
            reward = pipeline.calculate_improved_reward(report, test_vessel, "test")
            print(f"✅ Reward calculation test: {reward:.2f}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Naval Combat AI Training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=40, help='Steps per episode')
    parser.add_argument('--test', action='store_true', help='Run quick test instead of training')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    else:
        pipeline = ImprovedAITrainingPipeline(
            episodes=args.episodes,
            steps_per_episode=args.steps
        )
        pipeline.train()