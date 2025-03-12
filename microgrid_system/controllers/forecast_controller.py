import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

class ForecastController:
    """
    Controller that uses forecasting to predict future solar, load, and price values,
    then makes decisions based on these forecasts.
    """
    def __init__(self, forecast_horizon=24, prediction_window=8):
        self.forecast_horizon = forecast_horizon  # Hours to forecast ahead
        self.prediction_window = prediction_window  # Hours to use for prediction
        self.solar_model = RandomForestRegressor(n_estimators=100)
        self.load_model = RandomForestRegressor(n_estimators=100)
        self.price_model = RandomForestRegressor(n_estimators=100)
        self.solar_scaler = StandardScaler()
        self.load_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()  # Single scaler for input features
        self.history = []  # Stores past observations
        self.trained = False
    
    def collect_data(self, observation):
        """Add observation to history"""
        self.history.append(observation)
        
    def create_features(self, data):
        """Create features for prediction models"""
        features = []
        targets_solar = []
        targets_load = []
        targets_price = []
        
        # Minimum data needed to create features
        if len(data) < self.prediction_window + self.forecast_horizon:
            return None, None, None, None
        
        for i in range(len(data) - self.prediction_window - self.forecast_horizon + 1):
            # Features are the last prediction_window observations
            window = data[i:i+self.prediction_window]
            features.append(np.array(window).flatten())
            
            # Targets are the values forecast_horizon steps ahead
            target_idx = i + self.prediction_window + self.forecast_horizon - 1
            targets_solar.append(data[target_idx][1])  # Solar value
            targets_load.append(data[target_idx][2])   # Load value
            targets_price.append(data[target_idx][3])  # Price value
            
        return (np.array(features), 
                np.array(targets_solar), 
                np.array(targets_load), 
                np.array(targets_price))
    
    def train(self):
        """Train prediction models on collected data"""
        if len(self.history) < self.prediction_window + self.forecast_horizon:
            print(f"Not enough data for training: {len(self.history)} points")
            return False
        
        X, y_solar, y_load, y_price = self.create_features(self.history)
        
        if X is None:
            return False
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X)
        y_solar_scaled = self.solar_scaler.fit_transform(y_solar.reshape(-1, 1)).flatten()
        y_load_scaled = self.load_scaler.fit_transform(y_load.reshape(-1, 1)).flatten()
        y_price_scaled = self.price_scaler.fit_transform(y_price.reshape(-1, 1)).flatten()
        
        # Train models
        self.solar_model.fit(X_scaled, y_solar_scaled)
        self.load_model.fit(X_scaled, y_load_scaled)
        self.price_model.fit(X_scaled, y_price_scaled)
        
        self.trained = True
        print("Models trained successfully")
        return True
    
    def forecast(self):
        """Make forecasts using the trained models"""
        if not self.trained:
            return None, None, None
        
        if len(self.history) < self.prediction_window:
            return None, None, None
        
        # Create feature from most recent data
        recent_data = self.history[-self.prediction_window:]
        X = np.array(recent_data).flatten().reshape(1, -1)
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        solar_pred_scaled = self.solar_model.predict(X_scaled)
        load_pred_scaled = self.load_model.predict(X_scaled)
        price_pred_scaled = self.price_model.predict(X_scaled)
        
        # Inverse transform to get actual predictions
        solar_pred = self.solar_scaler.inverse_transform(solar_pred_scaled.reshape(-1, 1)).flatten()[0]
        load_pred = self.load_scaler.inverse_transform(load_pred_scaled.reshape(-1, 1)).flatten()[0]
        price_pred = self.price_scaler.inverse_transform(price_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        return solar_pred, load_pred, price_pred
    
    def predict(self, observation, deterministic=True):
        """
        Make a decision based on current observation and forecasts.
        
        Args:
            observation: Current state [hour_of_day, solar, load, price, battery_charge]
            deterministic: Whether prediction should be deterministic
            
        Returns:
            action: Battery charge/discharge rate
        """
        # Add observation to history
        self.collect_data(observation)
        
        # Extract current state
        hour_of_day = observation[0]
        solar = observation[1]
        load = observation[2]
        price = observation[3]
        battery_charge = observation[4]
        
        # Try to train if not yet trained
        if not self.trained and len(self.history) >= self.prediction_window + self.forecast_horizon:
            self.train()
        
        # Make forecasts if model is trained
        if self.trained:
            solar_pred, load_pred, price_pred = self.forecast()
            
            # If forecast successful, make decision based on predictions
            if solar_pred is not None:
                # Calculate net power (current and predicted)
                net_power_current = solar - load
                net_power_pred = solar_pred - load_pred
                
                # Logic based on forecasts
                if price_pred > price and battery_charge > 0.1:
                    # Price expected to rise, discharge battery
                    return np.array([-0.8])
                elif price_pred < price and battery_charge < 0.9:
                    # Price expected to fall, charge battery
                    return np.array([0.8])
                elif net_power_current > 0 and battery_charge < 0.95:
                    # Excess solar now, charge battery
                    return np.array([0.6])
                elif net_power_pred < 0 and battery_charge > 0.1:
                    # Deficit expected soon, preserve battery
                    return np.array([-0.3])
                else:
                    # Default - small charge if excess, small discharge if deficit
                    return np.array([0.2 if net_power_current > 0 else -0.2])
        
        # Default behavior if no trained model or insufficient data
        # Simple rule-based fallback
        if solar > load and battery_charge < 0.95:
            return np.array([0.5])  # Charge
        elif solar < load and battery_charge > 0.05:
            return np.array([-0.5])  # Discharge
        else:
            return np.array([0.0])  # Do nothing
    
    def save_models(self, save_dir="microgrid_system/results/models"):
        """Save trained models"""
        if not self.trained:
            print("Models not trained, nothing to save")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Implement model saving logic here
        # For example using joblib.dump() or pickle
        print(f"Models would be saved to {save_dir}")
    
    def load_models(self, load_dir="microgrid_system/results/models"):
        """Load trained models"""
        # Implement model loading logic here
        print(f"Models would be loaded from {load_dir}")
        self.trained = True 