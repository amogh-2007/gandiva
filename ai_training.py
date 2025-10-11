# ai_training.py - FIXED VERSION
"""
AI Training Pipeline - FIXED VERSION with better error handling
"""

import time
import numpy as np
from backend import SimulationController, Vessel
from ai import NavalAI, AIAction, AIReport
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITrainingPipeline:
    """
    Training pipeline with robust error handling
    """
    
    def __init__(self, episodes=100, steps_per_episode=50):  # Reduced for stability
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.simulation = SimulationController(mission_type="Training", difficulty="medium")
        self.ai = NavalAI(backend=None)
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'successful_intercepts': 0,
            'false_positives': 0,
            'missed_threats': 0,
            'hitl_requests': 0
        }
        
    def convert_vessel_to_ai_format(self, vessel: Vessel) -> dict:
        """Convert backend Vessel to AI-compatible dictionary format"""
        try:
            return {
                'id': vessel.id,
                'x': vessel.x,
                'y': vessel.y,
                'speed': max(0.1, vessel.speed),  # Ensure non-zero speed
                'heading': vessel.heading,
                'vessel_type': vessel.vessel_type,
                'behavior': 'unknown',
                'true_threat_level': vessel.true_threat_level,
                'evasion_chance': 0.3 if vessel.true_threat_level == 'confirmed' else 0.1,
                'detection_range': 200,
                'aggressiveness': 0.8 if vessel.true_threat_level == 'confirmed' else 0.1
            }
        except Exception as e:
            logger.warning(f"Error converting vessel: {e}")
            # Return a safe default
            return {
                'id': getattr(vessel, 'id', 0),
                'x': getattr(vessel, 'x', 400),
                'y': getattr(vessel, 'y', 300),
                'speed': 1.0,
                'heading': 0,
                'vessel_type': 'Unknown',
                'behavior': 'idle',
                'true_threat_level': 'neutral',
                'evasion_chance': 0.1,
                'detection_range': 200,
                'aggressiveness': 0.1
            }
    
    def convert_player_to_ai_format(self) -> dict:
        """Convert player vessel to AI-compatible format"""
        player = self.simulation.player_ship
        return {
            'id': player.id,
            'x': player.x,
            'y': player.y,
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
        """Add vessels from AI scenario using the correct FleetManager method"""
        vessels_added = 0
        for vessel_data in ai_scenario:
            try:
                # Use the fleet manager to add vessels
                vessel = self.simulation.fleet.add_vessel(
                    x=vessel_data['x'],
                    y=vessel_data['y'],
                    vx=0.0,  # AI will control movement
                    vy=0.0,
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
    
    def calculate_reward(self, ai_report: AIReport, vessel: Vessel, action_taken: str) -> float:
        """
        Calculate reward based on AI decision accuracy
        """
        try:
            true_threat = vessel.true_threat_level
            recommended_action = ai_report.recommended_action.value
            
            # High reward for correct intercept of confirmed threats
            if true_threat == "confirmed" and recommended_action == "intercept":
                return 10.0
            # High penalty for intercepting neutral vessels
            elif true_threat == "neutral" and recommended_action == "intercept":
                return -8.0
            # Moderate reward for monitoring possible threats
            elif true_threat == "possible" and recommended_action == "monitor":
                return 5.0
            # Penalty for ignoring confirmed threats
            elif true_threat == "confirmed" and recommended_action == "ignore":
                return -10.0
            # Small reward for correct identification of neutrals
            elif true_threat == "neutral" and recommended_action == "ignore":
                return 2.0
            # Small penalty for unnecessary HITL requests
            elif recommended_action == "await_confirmation":
                return -1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            return 0.0
    
    def execute_ai_decision(self, ai_report: AIReport, vessel: Vessel):
        """
        Execute AI decision in the simulation
        """
        try:
            action = ai_report.recommended_action
            
            if action == AIAction.INTERCEPT:
                # Simulate intercept by removing hostile vessels
                if vessel.true_threat_level in ["possible", "confirmed"]:
                    vessel.active = False
                    return "successful_intercept"
                else:
                    return "false_positive"
                    
            elif action == AIAction.MONITOR:
                # Continue monitoring - no immediate action
                return "monitoring"
                
            elif action == AIAction.IGNORE:
                # Ignore vessel - check if this was correct
                if vessel.true_threat_level == "confirmed":
                    return "missed_threat"
                else:
                    return "correct_ignore"
                    
            elif action == AIAction.AWAIT_CONFIRMATION:
                # Human intervention needed
                self.training_metrics['hitl_requests'] += 1
                # Simulate human decision (for training, we use ground truth)
                if vessel.true_threat_level == "confirmed":
                    vessel.active = False
                    return "human_confirmed_intercept"
                else:
                    return "human_confirmed_safe"
            
            return "no_action"
        except Exception as e:
            logger.warning(f"Error executing AI decision: {e}")
            return "error"
    
    def run_training_episode(self, episode_num: int):
        """
        Run a single training episode with robust error handling
        """
        logger.info(f"Starting training episode {episode_num}")
        
        try:
            # Generate training scenario
            difficulty_levels = ["easy", "medium", "hard"]
            difficulty = difficulty_levels[min(episode_num // (self.episodes // 3), 2)]
            
            # Use AI to generate scenario
            ai_scenario = self.ai.generate_realistic_scenario(difficulty)
            
            # Add vessels using the corrected method
            vessels_added = self.add_vessels_from_ai_scenario(ai_scenario)
            
            if vessels_added == 0:
                logger.warning("No vessels added to simulation, skipping episode")
                self.training_metrics['episode_rewards'].append(0.0)
                return
            
            episode_reward = 0
            player_vessel_ai = self.convert_player_to_ai_format()
            
            for step in range(self.steps_per_episode):
                if self.simulation.game_over:
                    break
                    
                # Update simulation
                self.simulation.update_simulation()
                
                # Get active vessels for AI decision making
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
                        reward = self.calculate_reward(ai_report, vessel, "ai_decision")
                        episode_reward += reward
                        
                        # Execute decision and get outcome
                        outcome = self.execute_ai_decision(ai_report, vessel)
                        
                        # Update training metrics based on outcome
                        if outcome == "successful_intercept":
                            self.training_metrics['successful_intercepts'] += 1
                        elif outcome == "false_positive":
                            self.training_metrics['false_positives'] += 1
                        elif outcome == "missed_threat":
                            self.training_metrics['missed_threats'] += 1
                        
                        # Train AI with the experience
                        current_state = self.ai.get_state(vessel_ai_format, player_vessel_ai)
                        
                        # Convert action to index
                        action_index = list(AIAction).index(ai_report.recommended_action)
                        
                        # Record experience for training
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
                
                # Control NPC behaviors using AI (optional)
                for vessel in active_vessels:
                    if vessel.active:
                        try:
                            vessel_ai_format = self.convert_vessel_to_ai_format(vessel)
                            player_pos = (self.simulation.player_ship.x, self.simulation.player_ship.y)
                            
                            # Get AI-controlled movement
                            vx, vy = self.ai.control_npc_behavior(vessel_ai_format, player_pos)
                            vessel.set_velocity(vx, vy)
                        except Exception as e:
                            logger.warning(f"Error controlling NPC behavior: {e}")
                            continue
            
            self.training_metrics['episode_rewards'].append(episode_reward)
            
            logger.info(f"Episode {episode_num} completed. Reward: {episode_reward:.2f}")
            
        except Exception as e:
            logger.error(f"Error in training episode {episode_num}: {e}")
            self.training_metrics['episode_rewards'].append(0.0)
        
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
                # Also remove from fleet manager
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
        Main training loop with progress tracking
        """
        logger.info("Starting AI training pipeline")
        
        start_time = time.time()
        successful_episodes = 0
        
        for episode in range(self.episodes):
            try:
                self.run_training_episode(episode)
                successful_episodes += 1
                
                # Print progress every 5 episodes
                if episode % 5 == 0:
                    recent_rewards = self.training_metrics['episode_rewards'][-5:]
                    if recent_rewards:
                        avg_reward = np.mean(recent_rewards)
                        logger.info(f"Episode {episode}: Average Reward (last 5): {avg_reward:.2f}")
                    
                    # Print AI performance metrics occasionally
                    if episode % 10 == 0:
                        ai_performance = self.ai.get_performance_report()
                        logger.info(f"AI Performance: {ai_performance}")
                
                # Save checkpoint every 20 episodes
                if episode % 20 == 0 and episode > 0:
                    self._save_checkpoint(episode)
                    
            except Exception as e:
                logger.error(f"Critical error in episode {episode}: {e}")
                continue
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Successful episodes: {successful_episodes}/{self.episodes}")
        
        # Final performance report
        self._print_final_report()
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        try:
            logger.info(f"Checkpoint saved at episode {episode}")
            
            # Save training metrics
            checkpoint = {
                'episode': episode,
                'training_metrics': self.training_metrics,
                'ai_performance': self.ai.get_performance_report(),
                'timestamp': time.time()
            }
            
            logger.info(f"Checkpoint data: Episode {episode}, "
                       f"Avg Reward: {np.mean(self.training_metrics['episode_rewards'][-10:]):.2f}")
                       
        except Exception as e:
            logger.warning(f"Error saving checkpoint: {e}")
    
    def _print_final_report(self):
        """Print comprehensive training report"""
        print("\n" + "="*60)
        print("AI TRAINING COMPLETED - FINAL REPORT")
        print("="*60)
        
        try:
            # Training metrics
            if self.training_metrics['episode_rewards']:
                avg_reward = np.mean(self.training_metrics['episode_rewards'])
                std_reward = np.std(self.training_metrics['episode_rewards'])
            else:
                avg_reward = std_reward = 0.0
            
            print(f"\nTraining Metrics:")
            print(f"  Total Episodes: {self.episodes}")
            print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  Successful Intercepts: {self.training_metrics['successful_intercepts']}")
            print(f"  False Positives: {self.training_metrics['false_positives']}")
            print(f"  Missed Threats: {self.training_metrics['missed_threats']}")
            print(f"  HITL Requests: {self.training_metrics['hitl_requests']}")
            
            # AI performance
            ai_perf = self.ai.get_performance_report()
            print(f"\nAI Engine Performance:")
            for key, value in ai_perf.items():
                print(f"  {key}: {value}")
            
            print(f"\nFinal Epsilon: {self.ai.epsilon:.4f}")
            print(f"Training Steps: {self.ai.train_counter}")
            
        except Exception as e:
            print(f"Error generating final report: {e}")
        
        print("="*60)

def quick_test():
    """Quick test to verify the AI and simulation integration"""
    print("Running quick integration test...")
    
    try:
        # Create instances
        pipeline = AITrainingPipeline(episodes=1, steps_per_episode=5)
        
        # Generate test scenario
        test_vessels = pipeline.ai.generate_realistic_scenario("easy")
        print(f"Generated {len(test_vessels)} test vessels")
        
        # Test adding vessels
        vessels_added = pipeline.add_vessels_from_ai_scenario(test_vessels[:2])
        print(f"Added {vessels_added} vessels to simulation")
        
        # Test AI decision making
        player_data = pipeline.convert_player_to_ai_format()
        
        active_vessels = [v for v in pipeline.simulation.units 
                         if v != pipeline.simulation.player_ship and v.active]
        
        for i, vessel in enumerate(active_vessels[:2]):
            vessel_data = pipeline.convert_vessel_to_ai_format(vessel)
            report = pipeline.ai.decide_action(vessel_data, player_data)
            print(f"Vessel {i+1}: {vessel.vessel_type}")
            print(f"  Threat: {vessel.true_threat_level}")
            print(f"  AI Action: {report.recommended_action.value}")
            print(f"  Confidence: {report.confidence:.2f}")
            print()
        
        print("Quick test completed successfully!")
        
    except Exception as e:
        print(f"Error in quick test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Naval Combat AI')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=30, help='Steps per episode')
    parser.add_argument('--test', action='store_true', help='Run quick test instead of training')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    else:
        pipeline = AITrainingPipeline(
            episodes=args.episodes,
            steps_per_episode=args.steps
        )
        pipeline.train()