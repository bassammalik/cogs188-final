import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MicrogridGymEnv(gym.Env):
    """
    Gym wrapper for the microgrid environment to make it compatible with Stable Baselines3.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, microgrid_env):
        super(MicrogridGymEnv, self).__init__()
        
        self.env = microgrid_env
        
        # Define action and observation space
        # Action space: continuous values in range [-1, 1] where:
        # -1 = maximum discharge rate
        # 1 = maximum charge rate
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [hour_of_day, solar, load, price, battery_charge]
        # Each with appropriate ranges
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),  # Minimum values
            high=np.array([23, 10, 10, 1, 1]),  # Maximum values 
            dtype=np.float32
        )
    
    def step(self, action):
        """
        Take a step in the environment using the provided action.
        
        Args:
            action: A continuous value in [-1, 1] representing battery charge/discharge
            
        Returns:
            observation: The new state
            reward: The reward for taking the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Extract the scalar value from the action array
        action_value = float(action[0])
        
        # Take a step in the microgrid environment
        observation, reward, done, truncated, info = self.env.step(action_value)
        
        return observation, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation: The initial state
        """
        return self.env.reset(**kwargs)
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: The rendering mode
        """
        return self.env.render() 