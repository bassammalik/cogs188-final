import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import os

class MicrogridEnv(gym.Env):
    """
    A simplified microgrid environment with synthetic data patterns.
    Added features:
    - Battery degradation modeling
    - Weather uncertainty with cloud events
    """
    def __init__(self, days=30, capacity=100, efficiency=0.9, 
                 enable_degradation=True, enable_weather_uncertainty=True):
        super(MicrogridEnv, self).__init__()
        
        # Environment parameters
        self.hours_per_day = 24
        self.total_hours = days * self.hours_per_day
        self.current_hour = 0
        self.battery_capacity = capacity  # kWh
        self.initial_capacity = capacity  # Store initial capacity for degradation tracking
        self.battery_charge = 0.5 * capacity  # Start at 50%
        self.efficiency = efficiency  # Battery round-trip efficiency
        
        # Feature flags to enable/disable new features
        self.enable_degradation = enable_degradation
        self.enable_weather_uncertainty = enable_weather_uncertainty
        
        # Battery degradation parameters
        self.cycle_count = 0
        self.partial_cycle = 0
        self.degradation_per_full_cycle = 0.0005  # 0.05% capacity loss per full cycle
        self.max_dod_experienced = 0.0  # Track deepest discharge
        self.calendar_degradation_daily = 0.00015  # 0.015% capacity loss per day
        self.days_elapsed = 0
        
        # Weather uncertainty parameters
        self.weather_forecast_error_stdev = 0.1  # Standard deviation of forecast errors (10%)
        self.cloud_event_probability = 0.05  # 5% chance of sudden cloud event per hour
        self.cloud_coverage_impact = 0.7  # Reduces solar output to 30%
        self.cloud_event_duration = 3  # Hours
        self.current_cloud_event = 0  # Duration counter for active cloud events
        
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
        
        # Generate forecast solar data (with increasing error by forecast horizon)
        if self.enable_weather_uncertainty:
            self.forecast_solar_data = np.zeros((self.total_hours, 24))  # 24-hour forecasts for each hour
            for h in range(self.total_hours):
                for f in range(24):  # Generate 24h of forecasts
                    if h+f < self.total_hours:
                        # Error increases with forecast horizon
                        error_factor = 1.0 + np.random.normal(0, self.weather_forecast_error_stdev * (1 + 0.05*f))
                        self.forecast_solar_data[h, f] = self.solar_data[h+f] * max(0, error_factor)
    
    def step(self, action):
        """Execute one time step of the environment."""
        # Get current data
        hour = self.current_hour
        load = self.load_data[hour]
        price = self.price_data[hour]
        
        # Apply weather uncertainties if enabled
        if self.enable_weather_uncertainty and self.current_cloud_event > 0:
            # Ongoing cloud event
            self.current_cloud_event -= 1
            solar = self.solar_data[hour] * (1 - self.cloud_coverage_impact)
        elif self.enable_weather_uncertainty and np.random.random() < self.cloud_event_probability:
            # New cloud event
            self.current_cloud_event = self.cloud_event_duration - 1
            solar = self.solar_data[hour] * (1 - self.cloud_coverage_impact)
        else:
            # Normal weather
            solar = self.solar_data[hour]
        
        # Track battery state before action for degradation calculation
        if self.enable_degradation:
            old_soc = self.battery_charge / self.battery_capacity  # State of Charge before action
        
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
        
        # Apply battery degradation if enabled
        degradation_info = {}
        if self.enable_degradation:
            # Calculate new state of charge
            new_soc = self.battery_charge / self.battery_capacity
            
            # Track battery usage for degradation calculation
            if (old_soc - new_soc) > 0:  # Discharge
                self.partial_cycle += (old_soc - new_soc)
                self.max_dod_experienced = max(self.max_dod_experienced, 1 - new_soc)
            else:  # Charge
                self.partial_cycle += (new_soc - old_soc) * 0.5  # Charging counts as half a cycle
            
            # Update cycle count and apply degradation if we've accumulated a full cycle
            if self.partial_cycle >= 1.0:
                full_cycles = int(self.partial_cycle)
                self.cycle_count += full_cycles
                self.partial_cycle -= full_cycles
                
                # Calculate degradation (higher for deeper discharges)
                dod_factor = 1.0 + (self.max_dod_experienced * 0.5)  # DoD penalty
                capacity_loss = full_cycles * self.degradation_per_full_cycle * dod_factor
                self.battery_capacity *= (1 - capacity_loss)
                self.max_dod_experienced = 0.0  # Reset after applying
                
            # Calendar aging - apply once per day
            if (self.current_hour % self.hours_per_day) == 0 and self.current_hour > 0:
                self.days_elapsed += 1
                self.battery_capacity *= (1 - self.calendar_degradation_daily)
            
            # Store degradation information
            degradation_info = {
                'battery_capacity': self.battery_capacity,
                'capacity_degradation_percent': 100 * (1 - self.battery_capacity / self.initial_capacity),
                'cycle_count': self.cycle_count,
                'partial_cycle': self.partial_cycle
            }
        
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
        
        # Add weather uncertainty info if enabled
        if self.enable_weather_uncertainty:
            # Add solar forecast for next 24 hours (or until end)
            forecast_horizon = min(24, self.total_hours - self.current_hour)
            forecast_data = np.zeros(forecast_horizon)
            for i in range(forecast_horizon):
                if self.current_hour + i < self.total_hours:
                    forecast_data[i] = self.forecast_solar_data[hour, i]
            
            info['solar_forecast'] = forecast_data
            info['is_cloudy'] = self.current_cloud_event > 0
            info['original_solar'] = self.solar_data[hour]  # The value without cloud cover
        
        # Add battery degradation info if enabled
        if self.enable_degradation:
            info.update(degradation_info)
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_hour = 0
        
        # Reset battery to initial conditions
        if self.enable_degradation:
            # Reset degradation parameters for a new episode
            self.battery_capacity = self.initial_capacity
            self.cycle_count = 0
            self.partial_cycle = 0
            self.max_dod_experienced = 0.0
            self.days_elapsed = 0
        
        self.battery_charge = 0.5 * self.battery_capacity
        
        # Reset weather conditions
        if self.enable_weather_uncertainty:
            self.current_cloud_event = 0
        
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
        print(f"  Battery: {self.battery_charge:.2f}/{self.battery_capacity:.2f} kWh")
        
        if self.enable_degradation:
            degradation = 100 * (1 - self.battery_capacity / self.initial_capacity)
            print(f"  Battery Health: {100-degradation:.2f}% (Cycles: {self.cycle_count:.1f})")
        
        if self.enable_weather_uncertainty and self.current_cloud_event > 0:
            print(f"  Weather: Cloudy (remaining: {self.current_cloud_event} hours)")
    
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
        
    def plot_battery_degradation(self, save_path="microgrid_system/results/battery_degradation.png"):
        """Plot battery degradation over time (call after simulation)."""
        if not self.enable_degradation:
            print("Battery degradation is not enabled.")
            return
            
        # Create a simplified simulation to show degradation over time
        days = 365 * 2  # 2 years
        cycles_per_day = 1.0  # 1 full cycle per day
        
        # Calculate degradation
        capacity = np.zeros(days)
        capacity[0] = 100  # Start at 100%
        
        for day in range(1, days):
            # Cycle degradation
            dod_factor = 1.0 + (0.8 * 0.5)  # Assuming 80% depth of discharge
            cycle_loss = cycles_per_day * self.degradation_per_full_cycle * dod_factor
            
            # Calendar degradation
            calendar_loss = self.calendar_degradation_daily
            
            # Combined degradation
            total_loss = cycle_loss + calendar_loss
            capacity[day] = capacity[day-1] * (1 - total_loss)
        
        # Plot degradation
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(days)/365, capacity)
        plt.xlabel('Years')
        plt.ylabel('Battery Capacity (%)')
        plt.title('Battery Capacity Degradation Over Time')
        plt.grid(True)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    def plot_weather_events(self, save_path="microgrid_system/results/weather_events.png"):
        """Run a simple simulation and plot weather events impact on solar generation."""
        if not self.enable_weather_uncertainty:
            print("Weather uncertainty is not enabled.")
            return
            
        # Simulate for a week
        days = 7
        hours = days * 24
        
        # Reset
        self.reset()
        
        # Storage for original and actual solar generation
        original_solar = np.zeros(hours)
        actual_solar = np.zeros(hours)
        cloudy_periods = np.zeros(hours)
        
        # Run through simulation
        for hour in range(hours):
            # Generate random action (just for simulation purposes)
            action = np.array([0.0])  # No battery action
            
            # Take step in environment
            _, _, _, _, info = self.step(action)
            
            # Store values
            original_solar[hour] = info['original_solar']
            actual_solar[hour] = info['solar']
            cloudy_periods[hour] = 1 if info['is_cloudy'] else 0
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot solar generation
        plt.plot(np.arange(hours)/24, original_solar, 'orange', label='Expected Solar')
        plt.plot(np.arange(hours)/24, actual_solar, 'r', label='Actual Solar')
        
        # Highlight cloudy periods
        cloudy_periods_idx = np.where(cloudy_periods == 1)[0]
        for idx in cloudy_periods_idx:
            plt.axvspan(idx/24, (idx+1)/24, alpha=0.2, color='gray')
        
        plt.xlabel('Days')
        plt.ylabel('Solar Generation (kW)')
        plt.title('Impact of Cloud Events on Solar Generation')
        plt.legend()
        plt.grid(True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() 