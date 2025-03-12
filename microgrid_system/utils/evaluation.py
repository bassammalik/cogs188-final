import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

class ControllerEvaluator:
    """
    Class to evaluate and compare different microgrid controllers.
    """
    def __init__(self, microgrid_env, controllers=None, output_dir="microgrid_system/results"):
        """
        Initialize the evaluator.
        
        Args:
            microgrid_env: The microgrid environment
            controllers: Dict of controllers to evaluate {name: controller}
            output_dir: Directory to save results
        """
        self.env = microgrid_env
        self.controllers = controllers or {}
        self.output_dir = output_dir
        self.results = {}
        
    def add_controller(self, name, controller):
        """Add a controller to the evaluation"""
        self.controllers[name] = controller
        
    def evaluate_controller(self, controller_name, episodes=1, steps_per_episode=24*7, verbose=True):
        """
        Evaluate a single controller over multiple episodes.
        
        Args:
            controller_name: Name of the controller to evaluate
            episodes: Number of episodes to run
            steps_per_episode: Number of steps per episode
            verbose: Whether to print progress
            
        Returns:
            episode_metrics: Dict of metrics for each episode
        """
        if controller_name not in self.controllers:
            raise ValueError(f"Controller {controller_name} not found")
            
        controller = self.controllers[controller_name]
        episode_metrics = []
        
        for episode in range(episodes):
            # Reset environment
            observation, _ = self.env.reset()
            
            # Initialize metrics for this episode
            total_reward = 0
            total_cost = 0
            total_energy_bought = 0
            total_energy_sold = 0
            battery_soc_history = []
            action_history = []
            
            # Track additional metrics
            solar_history = []
            load_history = []
            price_history = []
            net_load_history = []  # Load - Solar
            
            if verbose:
                iterator = tqdm(range(steps_per_episode), desc=f"Episode {episode+1}/{episodes}")
            else:
                iterator = range(steps_per_episode)
                
            for step in iterator:
                # Get action from controller
                action = controller.predict(observation)
                
                # Take step in environment
                next_observation, reward, done, truncated, info = self.env.step(float(action[0]))
                
                # Update metrics
                total_reward += reward
                total_cost += info.get('cost', 0)
                total_energy_bought += info.get('energy_bought', 0)
                total_energy_sold += info.get('energy_sold', 0)
                
                # Track state variables
                battery_soc_history.append(observation[4])
                action_history.append(float(action[0]))
                solar_history.append(observation[1])
                load_history.append(observation[2])
                price_history.append(observation[3])
                net_load_history.append(observation[2] - observation[1])
                
                # Update observation
                observation = next_observation
                
                if done:
                    break
            
            # Calculate average metrics
            avg_battery_soc = np.mean(battery_soc_history)
            avg_cost_per_step = total_cost / steps_per_episode
            
            # Store metrics for this episode
            episode_data = {
                'total_reward': total_reward,
                'total_cost': total_cost,
                'avg_cost_per_step': avg_cost_per_step,
                'avg_battery_soc': avg_battery_soc,
                'total_energy_bought': total_energy_bought,
                'total_energy_sold': total_energy_sold,
                'battery_soc_history': battery_soc_history,
                'action_history': action_history,
                'solar_history': solar_history,
                'load_history': load_history,
                'price_history': price_history,
                'net_load_history': net_load_history
            }
            
            episode_metrics.append(episode_data)
            
            if verbose:
                print(f"Episode {episode+1} - Reward: {total_reward:.2f}, Cost: {total_cost:.2f}")
                
        # Store results for this controller
        self.results[controller_name] = episode_metrics
        
        return episode_metrics
    
    def evaluate_all_controllers(self, episodes=1, steps_per_episode=24*7, verbose=True):
        """
        Evaluate all controllers.
        
        Args:
            episodes: Number of episodes per controller
            steps_per_episode: Number of steps per episode
            verbose: Whether to print progress
            
        Returns:
            all_results: Dict of results for each controller
        """
        if not self.controllers:
            raise ValueError("No controllers to evaluate")
            
        for name in self.controllers:
            print(f"\nEvaluating controller: {name}")
            self.evaluate_controller(name, episodes, steps_per_episode, verbose)
            
        return self.results
    
    def summarize_results(self):
        """
        Summarize evaluation results across all controllers.
        
        Returns:
            summary_df: DataFrame with summary statistics
        """
        if not self.results:
            raise ValueError("No results to summarize")
            
        summary_data = []
        
        for controller_name, episodes in self.results.items():
            # Calculate average metrics across episodes
            avg_reward = np.mean([ep['total_reward'] for ep in episodes])
            avg_cost = np.mean([ep['total_cost'] for ep in episodes])
            avg_energy_bought = np.mean([ep['total_energy_bought'] for ep in episodes])
            avg_energy_sold = np.mean([ep['total_energy_sold'] for ep in episodes])
            avg_battery_soc = np.mean([ep['avg_battery_soc'] for ep in episodes])
            
            summary_data.append({
                'Controller': controller_name,
                'Avg Reward': avg_reward,
                'Avg Cost': avg_cost,
                'Avg Energy Bought': avg_energy_bought,
                'Avg Energy Sold': avg_energy_sold,
                'Avg Battery SoC': avg_battery_soc
            })
            
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def plot_comparison(self, metric='total_cost', save=True):
        """
        Plot comparison of controllers based on a specific metric.
        
        Args:
            metric: The metric to compare
            save: Whether to save the plot
        """
        if not self.results:
            raise ValueError("No results to plot")
            
        plt.figure(figsize=(10, 6))
        
        data = []
        for controller_name, episodes in self.results.items():
            for i, ep in enumerate(episodes):
                data.append({
                    'Controller': controller_name,
                    'Episode': i+1,
                    metric: ep[metric]
                })
                
        df = pd.DataFrame(data)
        
        # Create plot
        sns.barplot(x='Controller', y=metric, data=df)
        plt.title(f'Comparison of Controllers: {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f"{self.output_dir}/{metric}_comparison.png")
            
        plt.show()
    
    def plot_time_series(self, controller_name, episode=0, metrics=None, save=True):
        """
        Plot time series data for a specific controller and episode.
        
        Args:
            controller_name: Name of the controller
            episode: Episode index
            metrics: List of metrics to plot
            save: Whether to save the plot
        """
        if controller_name not in self.results:
            raise ValueError(f"Results for controller {controller_name} not found")
            
        if episode >= len(self.results[controller_name]):
            raise ValueError(f"Episode {episode} not found for controller {controller_name}")
            
        episode_data = self.results[controller_name][episode]
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['battery_soc_history', 'action_history', 'price_history']
            
        # Create subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), sharex=True)
        
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric not in episode_data:
                print(f"Warning: Metric {metric} not found in episode data")
                continue
                
            axes[i].plot(episode_data[metric])
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{metric} - {controller_name}")
            
        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f"{self.output_dir}/{controller_name}_episode{episode}_timeseries.png")
            
        plt.show()
    
    def plot_all_time_series(self, episode=0, save=True):
        """
        Plot time series comparison of all controllers for a specific episode.
        
        Args:
            episode: Episode index
            save: Whether to save the plot
        """
        if not self.results:
            raise ValueError("No results to plot")
            
        # Key metrics to plot
        metrics = ['battery_soc_history', 'action_history', 'solar_history', 'load_history', 'price_history']
        
        # Create subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3*len(metrics)), sharex=True)
        
        # Add each controller's data to each subplot
        for controller_name, episodes in self.results.items():
            if episode >= len(episodes):
                print(f"Warning: Episode {episode} not found for controller {controller_name}")
                continue
                
            episode_data = episodes[episode]
            
            for i, metric in enumerate(metrics):
                if metric not in episode_data:
                    print(f"Warning: Metric {metric} not found for controller {controller_name}")
                    continue
                    
                axes[i].plot(episode_data[metric], label=controller_name)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f"{metric}")
                axes[i].legend()
                
        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f"{self.output_dir}/all_controllers_episode{episode}_timeseries.png")
            
        plt.show()
        
    def save_results(self, filename="results.csv"):
        """Save summarized results to CSV"""
        if not self.results:
            raise ValueError("No results to save")
            
        summary_df = self.summarize_results()
        
        os.makedirs(self.output_dir, exist_ok=True)
        summary_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
        print(f"Results saved to {self.output_dir}/{filename}")
        
        return summary_df 