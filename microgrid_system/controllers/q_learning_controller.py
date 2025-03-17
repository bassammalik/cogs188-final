import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class QLearningController:
    """
    Q-learning controller for the microgrid environment.
    Uses temporal difference learning to find optimal policy.
    
    Features:
    - Discretizes continuous state space
    - Uses epsilon-greedy exploration
    - Learns Q-values through interaction with environment
    """
    def __init__(self, 
                 learning_rate=0.1, 
                 discount_factor=0.95, 
                 exploration_rate=0.2,
                 exploration_decay=0.999,
                 min_exploration_rate=0.01,
                 battery_bins=10,
                 price_bins=5,
                 solar_bins=5,
                 load_bins=5,
                 action_bins=5,
                 model_path=None):
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Discretization parameters
        self.battery_bins = battery_bins
        self.price_bins = price_bins
        self.solar_bins = solar_bins
        self.load_bins = load_bins
        self.action_bins = action_bins
        
        # State space discretization ranges
        self.hour_bins = 24  # 24 hours in a day
        self.battery_range = [0, 100]  # Will be updated on first observation
        self.price_range = [0.05, 0.30]  # Typical price range
        self.solar_range = [0, 100]  # Will be updated on first observation
        self.load_range = [0, 100]  # Will be updated on first observation
        
        # Action space discretization
        self.actions = np.linspace(-1.0, 1.0, self.action_bins)
        
        # Initialize Q-table
        self.q_table = {}
        
        # Training metrics
        self.training_rewards = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded Q-learning model from {model_path}")
    
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state tuple"""
        hour_of_day = int(observation[0])
        
        # Update ranges based on observation if needed
        self.battery_range[1] = max(self.battery_range[1], observation[4] * 1.2)
        self.solar_range[1] = max(self.solar_range[1], observation[1] * 1.2)
        self.load_range[1] = max(self.load_range[1], observation[2] * 1.2)
        
        # Discretize each component
        solar_bin = min(self.solar_bins - 1, 
                        int(self.solar_bins * (observation[1] - self.solar_range[0]) / 
                            (self.solar_range[1] - self.solar_range[0])))
        
        load_bin = min(self.load_bins - 1, 
                       int(self.load_bins * (observation[2] - self.load_range[0]) / 
                           (self.load_range[1] - self.load_range[0])))
        
        price_bin = min(self.price_bins - 1, 
                        int(self.price_bins * (observation[3] - self.price_range[0]) / 
                            (self.price_range[1] - self.price_range[0])))
        
        battery_bin = min(self.battery_bins - 1, 
                          int(self.battery_bins * (observation[4] - self.battery_range[0]) / 
                              (self.battery_range[1] - self.battery_range[0])))
        
        # Return state as a tuple (can be used as dictionary key)
        return (hour_of_day, solar_bin, load_bin, price_bin, battery_bin)
    
    def get_q_values(self, state):
        """Get Q-values for a given state"""
        if state not in self.q_table:
            # Initialize with zeros if state not seen before
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]
    
    def predict(self, observation, deterministic=False):
        """
        Select action based on current policy
        If deterministic=True, always choose best action (for evaluation)
        If deterministic=False, use epsilon-greedy policy (for training)
        """
        state = self.discretize_state(observation)
        q_values = self.get_q_values(state)
        
        if not deterministic and np.random.random() < self.exploration_rate:
            # Exploration: choose random action
            action_idx = np.random.randint(0, len(self.actions))
        else:
            # Exploitation: choose best action
            action_idx = np.argmax(q_values)
        
        # Return action as a numpy array (same format as other controllers)
        return np.array([self.actions[action_idx]])
    
    def update(self, observation, action, reward, next_observation, done):
        """Update Q-values using the Q-learning update rule"""
        # Convert to discrete states
        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)
        
        # Find action index
        action_idx = np.argmin(np.abs(self.actions - action[0]))
        
        # Get current Q-value
        q_values = self.get_q_values(state)
        current_q = q_values[action_idx]
        
        # Get max Q-value for next state
        next_q_values = self.get_q_values(next_state)
        max_next_q = np.max(next_q_values)
        
        # Q-learning update rule
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        q_values[action_idx] = new_q
        
        # Update exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Track rewards for monitoring
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
    
    def train(self, env, episodes=100, steps_per_episode=24*7, save_path=None):
        """Train the Q-learning controller"""
        print(f"Training Q-learning controller for {episodes} episodes...")
        
        for episode in range(episodes):
            observation, _ = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Select action
                action = self.predict(observation, deterministic=False)
                
                # Take action
                next_observation, reward, done, _, info = env.step(action)
                
                # Update Q-values
                self.update(observation, action, reward, next_observation, done)
                
                # Track rewards
                episode_reward += reward
                
                # Update observation
                observation = next_observation
                
                if done:
                    break
            
            # Track episode rewards
            self.training_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-10:])
                print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Exploration Rate: {self.exploration_rate:.4f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            
        return self.training_rewards
    
    def save_model(self, path):
        """Save Q-table and parameters to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'params': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'exploration_rate': self.exploration_rate,
                    'battery_range': self.battery_range,
                    'price_range': self.price_range,
                    'solar_range': self.solar_range,
                    'load_range': self.load_range,
                }
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load Q-table and parameters from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            params = data['params']
            self.learning_rate = params['learning_rate']
            self.discount_factor = params['discount_factor']
            self.exploration_rate = params['exploration_rate']
            self.battery_range = params['battery_range']
            self.price_range = params['price_range']
            self.solar_range = params['solar_range']
            self.load_range = params['load_range']
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_rewards)
        plt.title('Q-Learning Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 