#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Grid Energy Management - Experiment Runner

This script runs experiments to compare different control strategies
for managing a microgrid with solar generation, battery storage, and grid connection.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
from microgrid_system.environment import MicrogridEnv
from microgrid_system.controllers import RuleBasedController, RLController, ForecastController
from microgrid_system.models import MicrogridGymEnv
from microgrid_system.utils import ControllerEvaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Grid Energy Management Experiments")
    
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to simulate (default: 7)")
    parser.add_argument("--battery-capacity", type=float, default=10.0,
                       help="Battery capacity in kWh (default: 10.0)")
    parser.add_argument("--efficiency", type=float, default=0.9,
                       help="Battery charging/discharging efficiency (default: 0.9)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to run for each controller (default: 3)")
    parser.add_argument("--train-rl", action="store_true",
                       help="Train the RL controller (default: False)")
    parser.add_argument("--rl-timesteps", type=int, default=50000,
                       help="Number of timesteps to train RL controller (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="microgrid_system/results",
                       help="Directory to save results (default: microgrid_system/results)")
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the microgrid environment"""
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create environment
    env = MicrogridEnv(
        days=args.days,
        capacity=args.battery_capacity,
        efficiency=args.efficiency
    )
    
    # Generate synthetic data
    env.reset()
    
    # Display environment info
    print(f"Environment created with:")
    print(f"  - {args.days} days of simulation")
    print(f"  - {args.battery_capacity} kWh battery capacity")
    print(f"  - {args.efficiency} battery efficiency")
    
    return env

def setup_controllers(env, args):
    """Set up the controllers to evaluate"""
    controllers = {}
    
    # Rule-based controller
    rule_controller = RuleBasedController(
        low_price_threshold=0.12,
        high_price_threshold=0.18
    )
    controllers["Rule-Based"] = rule_controller
    
    # Forecast-based controller
    forecast_controller = ForecastController(
        forecast_horizon=24,
        prediction_window=8
    )
    controllers["Forecast"] = forecast_controller
    
    # RL controller
    if args.train_rl:
        # Create gym wrapper for the environment
        gym_env = MicrogridGymEnv(env)
        
        # Create and train RL controller
        rl_controller = RLController(gym_env)
        
        print(f"\nTraining RL controller for {args.rl_timesteps} timesteps...")
        rl_controller.train(
            total_timesteps=args.rl_timesteps,
            save_path=f"{args.output_dir}/models/rl_controller"
        )
        controllers["RL"] = rl_controller
    else:
        # Look for pre-trained model
        model_path = f"{args.output_dir}/models/rl_controller.zip"
        if os.path.exists(model_path):
            print(f"Loading pre-trained RL model from {model_path}")
            gym_env = MicrogridGymEnv(env)
            rl_controller = RLController(gym_env, model_path=model_path)
            rl_controller.load_model()
            controllers["RL"] = rl_controller
        else:
            print("No pre-trained RL model found and --train-rl not specified")
    
    return controllers

def run_experiments(env, controllers, args):
    """Run experiments with all controllers"""
    # Create evaluator
    evaluator = ControllerEvaluator(
        microgrid_env=env,
        controllers=controllers,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    print("\nEvaluating controllers...")
    evaluator.evaluate_all_controllers(
        episodes=args.episodes,
        steps_per_episode=24*args.days,  # 24 hours per day
        verbose=True
    )
    
    # Summarize results
    print("\nResults summary:")
    summary_df = evaluator.summarize_results()
    print(summary_df)
    
    # Plot comparisons
    print("\nGenerating plots...")
    evaluator.plot_comparison(metric='total_cost')
    evaluator.plot_comparison(metric='total_reward')
    evaluator.plot_all_time_series(episode=0)
    
    # Save results
    evaluator.save_results(filename="experiment_results.csv")
    
    return evaluator

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Setup environment
    env = setup_environment(args)
    
    # Visualize synthetic data
    env.visualize_data(save_path=f"{args.output_dir}/synthetic_data.png")
    
    # Setup controllers
    controllers = setup_controllers(env, args)
    
    # Run experiments
    evaluator = run_experiments(env, controllers, args)
    
    print(f"\nExperiments completed. Results saved to {args.output_dir}")
    
    return evaluator

if __name__ == "__main__":
    evaluator = main() 