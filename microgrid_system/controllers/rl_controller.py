import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class RLController:
    """
    Reinforcement Learning controller using PPO from Stable Baselines3.
    """
    def __init__(self, env, model_path=None):
        """
        Initialize the RL controller.
        
        Args:
            env: The microgrid environment
            model_path: Path to a pre-trained model (if None, a new model will be created)
        """
        self.env = env
        self.model_path = model_path
        self.model = None
        
    def train(self, total_timesteps=100000, save_path="microgrid_system/results/models/ppo_microgrid"):
        """
        Train the RL model.
        
        Args:
            total_timesteps: Number of timesteps to train for
            save_path: Path to save the trained model
        """
        # Create a dummy vectorized environment (required by Stable Baselines)
        env = DummyVecEnv([lambda: self.env])
        
        # Create and train the model
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        model.learn(total_timesteps=total_timesteps)
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        self.model = model
        print(f"Model saved to {save_path}")
        
    def load_model(self, model_path=None):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model
        """
        path = model_path if model_path is not None else self.model_path
        if path is None:
            raise ValueError("No model path provided")
        
        self.model = PPO.load(path)
        print(f"Model loaded from {path}")
        
    def predict(self, observation, deterministic=True):
        """
        Predict an action given an observation.
        
        Args:
            observation: The current state observation
            deterministic: Whether to return a deterministic action
            
        Returns:
            action: The predicted action
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action 