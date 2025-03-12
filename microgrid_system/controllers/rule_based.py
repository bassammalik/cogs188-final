import numpy as np

class RuleBasedController:
    """
    A simple rule-based controller for the microgrid.
    Charges when solar exceeds load or prices are low.
    Discharges when prices are high.
    """
    def __init__(self, low_price_threshold=0.12, high_price_threshold=0.18):
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
    
    def predict(self, observation, deterministic=True):
        # Extract state information
        hour_of_day = observation[0]
        solar = observation[1]
        load = observation[2]
        price = observation[3]
        battery_charge = observation[4]
        
        # Calculate net power (positive if excess solar, negative if deficit)
        net_power = solar - load
        
        # Define action based on simple rules
        if price <= self.low_price_threshold:
            # Low price period - charge battery if not full
            if battery_charge < 0.95:  # Less than 95% full
                return np.array([1.0])  # Charge at max rate
            else:
                return np.array([0.0])  # Do nothing
        elif price >= self.high_price_threshold:
            # High price period - discharge battery if not empty
            if battery_charge > 0.05:  # More than 5% charge
                return np.array([-1.0])  # Discharge at max rate
            else:
                return np.array([0.0])  # Do nothing
        else:
            # Normal price period
            if net_power > 0:
                # Excess solar - charge battery
                return np.array([0.5])  # Charge at half rate
            elif net_power < 0:
                # Deficit - slight discharge
                return np.array([-0.3])  # Discharge at 30% rate
            else:
                return np.array([0.0])  # Do nothing 