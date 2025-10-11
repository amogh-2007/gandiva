# curriculum_trainer.py
"""
Advanced AI Training with Curriculum Learning - FIXED VERSION
"""

import numpy as np
from ai_training import EnhancedAITrainingPipeline  # Changed import

class CurriculumAITrainer:
    """
    Implements curriculum learning for naval AI training - FIXED
    """
    
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'Beginner',
                'episodes': 20,  # Reduced for testing
                'difficulty': 'easy',
                'success_threshold': 0.7,
                'focus': 'basic_threat_detection'
            },
            {
                'name': 'Intermediate', 
                'episodes': 30,
                'difficulty': 'medium',
                'success_threshold': 0.6,
                'focus': 'mixed_scenarios'
            },
            {
                'name': 'Advanced',
                'episodes': 40,
                'difficulty': 'hard',
                'success_threshold': 0.5,
                'focus': 'complex_threats'
            }
        ]
    
    def calculate_stage_performance(self, pipeline: EnhancedAITrainingPipeline, stage_config: dict) -> float:  # Updated type hint
        """Calculate performance metric for current stage"""
        recent_rewards = pipeline.training_metrics['episode_rewards'][-5:]  # Smaller window
        if not recent_rewards:
            return 0.0
        
        avg_reward = np.mean(recent_rewards)
        max_possible = self._get_max_reward_for_difficulty(stage_config['difficulty'])
        
        return avg_reward / max_possible if max_possible > 0 else 0.0
    
    def _get_max_reward_for_difficulty(self, difficulty: str) -> float:
        """Estimate maximum possible reward for a difficulty level"""
        base_rewards = {
            'easy': 50.0,
            'medium': 80.0, 
            'hard': 120.0,
            'expert': 200.0
        }
        return base_rewards.get(difficulty, 50.0)
    
    def train_with_curriculum(self):
        """Main curriculum training loop"""
        print("Starting Curriculum Training")
        print("=" * 50)
        
        pipeline = EnhancedAITrainingPipeline(episodes=1, steps_per_episode=50)  # Updated class name
        
        total_episodes = 0
        
        for stage_num, stage_config in enumerate(self.curriculum_stages):
            print(f"\n🎯 Stage {stage_num + 1}: {stage_config['name']}")
            print(f"   Difficulty: {stage_config['difficulty']}")
            print(f"   Episodes: {stage_config['episodes']}")
            print(f"   Success Threshold: {stage_config['success_threshold']:.0%}")
            print("-" * 30)
            
            # Configure pipeline for this stage
            pipeline.episodes = stage_config['episodes']
            
            # Train for this stage
            stage_episodes = 0
            while stage_episodes < stage_config['episodes']:
                episodes_to_run = min(5, stage_config['episodes'] - stage_episodes)  # Smaller batches
                
                # Run batch of episodes
                for _ in range(episodes_to_run):
                    pipeline.run_training_episode(total_episodes)
                    total_episodes += 1
                    stage_episodes += 1
                
                # Check performance
                performance = self.calculate_stage_performance(pipeline, stage_config)
                print(f"   Progress: {stage_episodes}/{stage_config['episodes']} | "
                      f"Performance: {performance:.1%}")
                
                # Early advancement if performing well
                if (performance >= stage_config['success_threshold'] * 1.2 and 
                    stage_episodes >= stage_config['episodes'] * 0.5):
                    print("   ⚡ Early advancement to next stage!")
                    break
            
            # Final performance check for stage
            final_performance = self.calculate_stage_performance(pipeline, stage_config)
            if final_performance >= stage_config['success_threshold']:
                print(f"   ✅ Stage {stage_num + 1} completed successfully!")
            else:
                print(f"   ⚠️  Stage {stage_num + 1} completed below target")
            
            # Save checkpoint after each stage
            pipeline._save_checkpoint(total_episodes)
        
        print("\n" + "=" * 50)
        print("🎓 Curriculum Training Completed!")
        pipeline._print_comprehensive_report()  # Fixed method name

if __name__ == "__main__":
    # Choose training method
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curriculum":
        trainer = CurriculumAITrainer()
        trainer.train_with_curriculum()
    else:
        # Standard training with smaller defaults for testing
        pipeline = EnhancedAITrainingPipeline(episodes=50, steps_per_episode=50)  # Updated class name
        pipeline.train()