import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import os

class MicrogridEnv(gym.Env):
    """
    A simplified microgrid environment with synthetic data patterns.
    """
    def __init__(self, days=30, capacity=100, efficiency=0.9):
        super(MicrogridEnv, self).__init__()
        
        # Environment parameters
        self.hours_per_day = 24
        self.total_hours = days * self.hours_per_day
        self.current_hour = 0
        self.battery_capacity = capacity  # kWh
        self.battery_charge = 0.5 * capacity  # Start at 50%
        self.efficiency = efficiency  # Battery round-trip efficiency
        
        # Generate synthetic data
        self.generate_data(days)
        
        # Define action and observation spaces
        # Action: battery charge/discharge rate (-1 to 1, normalized)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: [hour_of_day, solar_power, load, price, battery_charge]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]), 
            high=np.array([23, np.inf, np.inf, np.inf, self.battery_capacity]), 
            dtype=np.float32
        )
    
    def generate_data(self, days):
        """Generate synthetic data for solar, load, and prices."""
        # Create time indices
        hours = np.arange(self.total_hours)
        hour_of_day = hours % self.hours_per_day
        day_of_week = (hours // self.hours_per_day) % 7
        
        # Generate solar data with daily pattern + noise
        # Solar only during daylight (7am-7pm)
        solar_pattern = np.zeros(self.hours_per_day)
        daylight_hours = np.arange(7, 19)  # 7am to 7pm
        solar_pattern[daylight_hours] = np.sin(np.pi * (daylight_hours - 7) / 12)
        
        # Repeat for all days and add variability
        self.solar_data = np.tile(solar_pattern, days)
        # Add day-to-day variability (more on weekends)
        for d in range(days):
            day_idx = d * self.hours_per_day
            # Base variability factor (0.7-1.3)
            variability = 0.7 + 0.6 * np.random.random()
            # Weekend adjustment
            if (d % 7) >= 5:  # weekend
                variability *= 1.2  # 20% more solar on weekends (clearer weather)
            self.solar_data[day_idx:day_idx+self.hours_per_day] *= 50 * variability
        
        # Add small random noise
        self.solar_data *= (0.9 + 0.2 * np.random.random(self.total_hours))
        
        # Generate load data (daily pattern + weekly pattern + noise)
        load_pattern = 20 + 15 * np.sin(np.pi * ((hour_of_day + 5) % self.hours_per_day) / 12)
        weekday_factor = np.ones(self.total_hours)
        
        # Reduce demand on weekends
        weekend_mask = (day_of_week >= 5)
        weekday_factor[weekend_mask] = 0.8
        
        # Calculate final load
        self.load_data = load_pattern * weekday_factor
        
        # Add random noise
        self.load_data *= (0.9 + 0.2 * np.random.random(self.total_hours))
        
        # Generate price data (daily pattern + weekly pattern + peaks)
        base_price = 0.10  # $0.10/kWh
        
        # Time-of-use pricing
        self.price_data = np.ones(self.total_hours) * base_price
        
        # Peak pricing (3pm-9pm on weekdays)
        peak_hours = (hour_of_day >= 15) & (hour_of_day < 21) & (day_of_week < 5)
        self.price_data[peak_hours] = 0.25  # $0.25/kWh during peak
        
        # Mid-peak (7am-3pm on weekdays)
        mid_hours = (hour_of_day >= 7) & (hour_of_day < 15) & (day_of_week < 5)
        self.price_data[mid_hours] = 0.15  # $0.15/kWh during mid-peak
        
        # Add random noise
        self.price_data *= (0.95 + 0.1 * np.random.random(self.total_hours))
    
    def step(self, action):
        """Execute one time step of the environment."""
        # Get current data
        hour = self.current_hour
        solar = self.solar_data[hour]
        load = self.load_data[hour]
        price = self.price_data[hour]
        
        # Normalize action to max charge/discharge rate
        max_rate = 0.2 * self.battery_capacity  # 20% per hour max
        battery_power = float(action) * max_rate
        
        # Apply efficiency losses
        if battery_power > 0:  # Charging
            battery_energy_change = battery_power * np.sqrt(self.efficiency)
        else:  # Discharging
            battery_energy_change = battery_power / np.sqrt(self.efficiency)
        
        # Update battery charge
        new_charge = self.battery_charge + battery_energy_change
        
        # Enforce battery capacity constraints
        if new_charge > self.battery_capacity:
            battery_energy_change = self.battery_capacity - self.battery_charge
            new_charge = self.battery_capacity
        elif new_charge < 0:
            battery_energy_change = -self.battery_charge
            new_charge = 0
        
        self.battery_charge = new_charge
        
        # Calculate net grid energy exchange
        grid_energy = load - solar - battery_energy_change
        
        # Calculate cost
        if grid_energy > 0:  # Buying from grid
            cost = grid_energy * price
        else:  # Selling to grid (at 80% of buying price)
            cost = grid_energy * price * 0.8
        
        # Calculate reward (negative cost)
        reward = -cost
        
        # Move to next hour
        self.current_hour += 1
        done = self.current_hour >= self.total_hours
        
        # Create observation
        observation = np.array([
            self.current_hour % self.hours_per_day,  # Hour of day (0-23)
            solar,
            load,
            price,
            self.battery_charge
        ], dtype=np.float32)
        
        # Store additional info
        info = {
            'cost': cost,
            'solar': solar,
            'load': load,
            'grid_energy': grid_energy,
            'battery_change': battery_energy_change,
            'energy_bought': max(0, grid_energy),
            'energy_sold': max(0, -grid_energy)
        }
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_hour = 0
        self.battery_charge = 0.5 * self.battery_capacity
        
        observation = np.array([
            0,  # Hour 0
            self.solar_data[0],
            self.load_data[0],
            self.price_data[0],
            self.battery_charge
        ], dtype=np.float32)
        
        return observation, {}
    
    def render(self):
        """Render the environment (simplified)."""
        hour = self.current_hour - 1
        print(f"Hour {hour} (Day {hour // 24 + 1}, Hour {hour % 24}):")
        print(f"  Solar: {self.solar_data[hour]:.2f} kW")
        print(f"  Load: {self.load_data[hour]:.2f} kW")
        print(f"  Price: ${self.price_data[hour]:.3f}/kWh")
        print(f"  Battery: {self.battery_charge:.2f}/{self.battery_capacity} kWh")
    
    def visualize_data(self, save_path="microgrid_system/results/synthetic_data.png"):
        """Generate and save a visualization of the synthetic data."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Time axis for first week only
        hours = np.arange(min(24*7, self.total_hours))
        days = hours / 24
        
        # Plot solar data
        axes[0].plot(days, self.solar_data[:len(hours)], 'orange')
        axes[0].set_ylabel('Solar Generation (kW)')
        axes[0].set_title('Microgrid Environment Data (First Week)')
        axes[0].grid(True)
        
        # Plot load data
        axes[1].plot(days, self.load_data[:len(hours)], 'blue')
        axes[1].set_ylabel('Building Load (kW)')
        axes[1].grid(True)
        
        # Plot price data
        axes[2].plot(days, self.price_data[:len(hours)], 'red')
        axes[2].set_ylabel('Electricity Price ($/kWh)')
        axes[2].set_xlabel('Days')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 