# curriculum_trainer.py
"""
Advanced AI Training with Curriculum Learning
Completes the AI training system architecture
"""

import numpy as np
from ai_training import EnhancedAITrainingPipeline

class CurriculumAITrainer:
    """
    Implements curriculum learning for progressive AI training
    Stages: Beginner -> Intermediate -> Advanced -> Expert
    """
    
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'Beginner',
                'episodes': 25,
                'difficulty': 'easy',
                'success_threshold': 0.65,
                'focus': 'basic_threat_detection',
                'min_accuracy': 0.70
            },
            {
                'name': 'Intermediate', 
                'episodes': 35,
                'difficulty': 'medium',
                'success_threshold': 0.60,
                'focus': 'mixed_scenarios',
                'min_accuracy': 0.65
            },
            {
                'name': 'Advanced',
                'episodes': 45,
                'difficulty': 'hard',
                'success_threshold': 0.55,
                'focus': 'complex_threats',
                'min_accuracy': 0.60
            },
            {
                'name': 'Expert',
                'episodes': 50,
                'difficulty': 'expert',
                'success_threshold': 0.50,
                'focus': 'elite_tactics',
                'min_accuracy': 0.55
            }
        ]
        
        self.stage_performance_history = []
    
    def calculate_stage_performance(self, pipeline: EnhancedAITrainingPipeline, stage_config: dict) -> dict:
        """Calculate comprehensive performance metrics for current stage"""
        if not pipeline.training_metrics['episode_rewards']:
            return {'score': 0.0, 'reward_score': 0.0, 'accuracy_score': 0.0}
        
        # FIXED: Handle empty metrics
        recent_rewards = pipeline.training_metrics['episode_rewards'][-10:] if len(pipeline.training_metrics['episode_rewards']) >= 10 else pipeline.training_metrics['episode_rewards']
        recent_accuracy = pipeline.training_metrics['episode_threat_accuracy'][-10:] if len(pipeline.training_metrics['episode_threat_accuracy']) >= 10 else pipeline.training_metrics['episode_threat_accuracy']
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.0
        
        # Normalize scores
        max_reward = self._get_max_reward_for_difficulty(stage_config['difficulty'])
        reward_score = min(avg_reward / max_reward, 1.0) if max_reward > 0 else 0.0
        accuracy_score = avg_accuracy
        
        # Combined score (weighted)
        combined_score = (reward_score * 0.6) + (accuracy_score * 0.4)
        
        return {
            'score': combined_score,
            'reward_score': reward_score,
            'accuracy_score': accuracy_score,
            'avg_reward': avg_reward,
            'avg_accuracy': avg_accuracy
        }
    
    def _get_max_reward_for_difficulty(self, difficulty: str) -> float:
        """Estimate maximum possible reward for a difficulty level"""
        base_rewards = {
            'easy': 60.0,
            'medium': 100.0, 
            'hard': 150.0,
            'expert': 220.0
        }
        return base_rewards.get(difficulty, 60.0)
    
    def train_with_curriculum(self):
        """Main curriculum training loop with progressive difficulty"""
        print("🎓 STARTING CURRICULUM TRAINING")
        print("=" * 55)
        
        pipeline = EnhancedAITrainingPipeline(episodes=1, steps_per_episode=50)
        total_episodes = 0
        
        for stage_num, stage_config in enumerate(self.curriculum_stages):
            print(f"\n🎯 STAGE {stage_num + 1}: {stage_config['name']}")
            print(f"   📊 Difficulty: {stage_config['difficulty']}")
            print(f"   🎯 Episodes: {stage_config['episodes']}")
            print(f"   🎯 Success Threshold: {stage_config['success_threshold']:.0%}")
            print(f"   🎯 Focus: {stage_config['focus']}")
            print("-" * 40)
            
            # Configure pipeline for this stage
            pipeline.episodes = stage_config['episodes']
            stage_performance = []
            
            # Train for this stage
            stage_episodes = 0
            max_retries = 2
            retry_count = 0
            
            while stage_episodes < stage_config['episodes'] and retry_count <= max_retries:
                episodes_to_run = min(5, stage_config['episodes'] - stage_episodes)
                
                # Run batch of episodes
                for _ in range(episodes_to_run):
                    pipeline.run_training_episode(total_episodes)
                    total_episodes += 1
                    stage_episodes += 1
                
                # Check performance
                performance = self.calculate_stage_performance(pipeline, stage_config)
                stage_performance.append(performance['score'])
                
                print(f"   📈 Progress: {stage_episodes:2d}/{stage_config['episodes']} | "
                      f"Score: {performance['score']:.1%} | "
                      f"Reward: {performance['avg_reward']:6.1f} | "
                      f"Accuracy: {performance['avg_accuracy']:5.1%}")
                
                # Early advancement if performing exceptionally well
                if (performance['score'] >= stage_config['success_threshold'] * 1.3 and 
                    stage_episodes >= stage_config['episodes'] * 0.6):
                    print("   ⚡ Early advancement - exceptional performance!")
                    break
            
            # Stage completion assessment
            final_performance = self.calculate_stage_performance(pipeline, stage_config)
            self.stage_performance_history.append(final_performance)
            
            if final_performance['score'] >= stage_config['success_threshold']:
                print(f"   ✅ Stage {stage_num + 1} completed successfully!")
                status = "PASS"
            else:
                print(f"   ⚠️  Stage {stage_num + 1} completed below target")
                print(f"   💡 Recommendation: More training needed for {stage_config['focus']}")
                status = "NEEDS_IMPROVEMENT"
                
                # FIXED: Limit retries to prevent infinite loops
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"   🔄 Retrying stage {stage_num + 1} (attempt {retry_count + 1})...")
                    stage_episodes = 0
                    continue
            
            # Stage summary
            print(f"   📋 Stage Summary:")
            print(f"      Final Score: {final_performance['score']:.1%}")
            print(f"      Avg Reward:  {final_performance['avg_reward']:7.1f}")
            print(f"      Accuracy:    {final_performance['avg_accuracy']:6.1%}")
            print(f"      Status:      {status}")
            
            # Save checkpoint after each stage
            pipeline._save_checkpoint(total_episodes)
            
            # Check if we should continue to next stage
            if status == "NEEDS_IMPROVEMENT" and stage_num < len(self.curriculum_stages) - 1:
                try:
                    retrain = input(f"   🔄 Retry Stage {stage_num + 1}? (y/n): ")
                    if retrain.lower() == 'y':
                        stage_num -= 1  # Repeat current stage
                        print("   🔄 Repeating stage with additional training...")
                except (EOFError, KeyboardInterrupt):
                    print("\n   ⏹️  Training interrupted by user")
                    break
        
        print("\n" + "=" * 55)
        print("🎓 CURRICULUM TRAINING COMPLETED!")
        self._print_curriculum_summary(pipeline)
    
    def _print_curriculum_summary(self, pipeline: EnhancedAITrainingPipeline):
        """Print comprehensive curriculum training summary"""
        print("\n" + "=" * 70)
        print("🎓 CURRICULUM TRAINING - COMPREHENSIVE SUMMARY")
        print("=" * 70)
        
        if not self.stage_performance_history:
            print("   No training data available")
            return
            
        print(f"\n📊 OVERALL PERFORMANCE ACROSS STAGES:")
        for i, (stage, performance) in enumerate(zip(self.curriculum_stages, self.stage_performance_history)):
            # FIXED: Handle case where performance history might be shorter
            if i < len(self.stage_performance_history):
                perf_data = self.stage_performance_history[i]
                status = "✅ PASS" if perf_data['score'] >= stage['success_threshold'] else "⚠️ NEEDS WORK"
                print(f"   Stage {i+1}: {stage['name']:12} | Score: {perf_data['score']:5.1%} | {status}")
            else:
                print(f"   Stage {i+1}: {stage['name']:12} | No data")
        
        # Overall statistics
        valid_scores = [p['score'] for p in self.stage_performance_history if p['score'] > 0]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
            
            print(f"\n📈 OVERALL STATISTICS:")
            print(f"   Average Stage Score: {avg_score:6.1%}")
            print(f"   Best Stage Score:    {max_score:6.1%}")
            print(f"   Worst Stage Score:   {min_score:6.1%}")
            
            if avg_score >= 0.7:
                rating = "🌟 ELITE"
            elif avg_score >= 0.6:
                rating = "🎯 ADVANCED"
            elif avg_score >= 0.5:
                rating = "📊 COMPETENT"
            else:
                rating = "📚 NEEDS TRAINING"
            
            print(f"   Overall Rating:      {rating}")
        else:
            print(f"\n📈 OVERALL STATISTICS: No valid performance data")
        
        # Final AI performance
        try:
            ai_perf = pipeline._get_ai_performance_report()
            print(f"\n🧠 FINAL AI PERFORMANCE:")
            print(f"   Total Training Steps: {ai_perf.get('training_steps', 0):6d}")
            print(f"   Final Epsilon:        {ai_perf.get('current_epsilon', 0.0):9.4f}")
            print(f"   Autonomous Rate:      {ai_perf.get('autonomous_success_rate', '0%'):>9}")
        except Exception as e:
            print(f"\n🧠 FINAL AI PERFORMANCE: Error retrieving data - {e}")
        
        print("=" * 70)

def main():
    """Main function for curriculum training"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curriculum":
        trainer = CurriculumAITrainer()
        trainer.train_with_curriculum()
    else:
        # Standard training
        print("🚀 Starting Standard AI Training...")
        print("💡 Use '--curriculum' flag for curriculum learning")
        pipeline = EnhancedAITrainingPipeline(episodes=50, steps_per_episode=50)
        pipeline.train()

if __name__ == "__main__":
    main()