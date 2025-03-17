import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class MonteCarloController:
    """
    Monte Carlo controller for the microgrid environment.
    Uses episode-based learning to find optimal policy.
    
    Features:
    - Discretizes continuous state space
    - Uses epsilon-greedy exploration
    - Learns state-action values through complete episodes
    - First-visit Monte Carlo policy evaluation
    """
    def __init__(self, 
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
        
        # Initialize value function
        self.q_values = {}  # State-action values
        self.returns = {}   # Returns for each state-action pair
        
        # Training metrics
        self.training_rewards = []
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded Monte Carlo model from {model_path}")
    
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
        if state not in self.q_values:
            # Initialize with zeros if state not seen before
            self.q_values[state] = np.zeros(len(self.actions))
        return self.q_values[state]
    
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
    
    def train(self, env, episodes=100, steps_per_episode=24*7, save_path=None):
        """Train the Monte Carlo controller using first-visit MC"""
        print(f"Training Monte Carlo controller for {episodes} episodes...")
        
        for episode in range(episodes):
            # Initialize episode data
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            # Reset environment
            observation, _ = env.reset()
            episode_reward = 0
            
            # Generate episode
            for step in range(steps_per_episode):
                # Select action
                action = self.predict(observation, deterministic=False)
                
                # Store state and action
                state = self.discretize_state(observation)
                action_idx = np.argmin(np.abs(self.actions - action[0]))
                
                episode_states.append(state)
                episode_actions.append(action_idx)
                
                # Take action
                next_observation, reward, done, _, info = env.step(action)
                
                # Store reward
                episode_rewards.append(reward)
                episode_reward += reward
                
                # Update observation
                observation = next_observation
                
                if done:
                    break
            
            # Process episode (first-visit Monte Carlo)
            G = 0  # Initialize return
            visited_state_actions = set()
            
            # Process episode in reverse order
            for t in range(len(episode_rewards) - 1, -1, -1):
                # Update return
                G = self.discount_factor * G + episode_rewards[t]
                
                # Get state and action
                state = episode_states[t]
                action_idx = episode_actions[t]
                
                # Check if this is the first visit to this state-action pair
                state_action = (state, action_idx)
                if state_action not in visited_state_actions:
                    # Mark as visited
                    visited_state_actions.add(state_action)
                    
                    # Update returns
                    if state_action not in self.returns:
                        self.returns[state_action] = []
                    self.returns[state_action].append(G)
                    
                    # Update Q-value (average of returns)
                    self.q_values[state][action_idx] = np.mean(self.returns[state_action])
            
            # Update exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * self.exploration_decay
            )
            
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
        """Save Q-values and parameters to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'q_values': self.q_values,
                'returns': self.returns,
                'params': {
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
        """Load Q-values and parameters from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_values = data['q_values']
            self.returns = data['returns']
            params = data['params']
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
        plt.title('Monte Carlo Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 