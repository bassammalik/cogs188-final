# Smart Grid Energy Management

This project implements a simplified smart grid energy management system using different control strategies to optimize battery usage for a microgrid with solar generation, load, and grid connection.

## Project Overview

The project simulates a microgrid with the following components:
- Solar PV generation (with daily patterns and variability)
- Building load (with daily patterns and variability)
- Battery storage system
- Grid connection with time-varying electricity prices

The goal is to minimize the cost of electricity by:
- Charging the battery when solar production exceeds load or when electricity prices are low
- Discharging the battery when load exceeds solar production or when electricity prices are high

## AI Methods Implemented

The project compares three different control strategies:

1. **Rule-Based Controller**: A simple heuristic controller that follows predefined rules based on current conditions (solar, load, price, battery state of charge).

2. **Reinforcement Learning Controller**: Uses Proximal Policy Optimization (PPO) to learn an optimal control policy through interaction with the environment.

3. **Forecast-Based Controller**: Uses machine learning to forecast future solar, load, and price values, then makes decisions based on these forecasts.

## Project Structure

```
microgrid_system/
├── __init__.py
├── environment.py              # Main environment simulation
├── controllers/                # Control strategies
│   ├── __init__.py
│   ├── rule_based.py           # Rule-based controller
│   ├── rl_controller.py        # Reinforcement learning controller
│   └── forecast_controller.py  # Forecast-based controller
├── models/                     # ML models and wrappers
│   ├── __init__.py
│   └── gym_wrapper.py          # Gym wrapper for RL
├── utils/                      # Utilities
│   ├── __init__.py
│   └── evaluation.py           # Evaluation framework
└── results/                    # Output directory
    └── __init__.py
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Gym
- Stable-Baselines3
- scikit-learn
- tqdm

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn gym stable-baselines3 scikit-learn tqdm
```

## Usage

Run the experiment with default settings:

```bash
python run_experiments.py
```

### Command-line options:

- `--days`: Number of days to simulate (default: 7)
- `--battery-capacity`: Battery capacity in kWh (default: 10.0)
- `--efficiency`: Battery charging/discharging efficiency (default: 0.9)
- `--episodes`: Number of episodes to run for each controller (default: 3)
- `--train-rl`: Train the RL controller (default: False)
- `--rl-timesteps`: Number of timesteps to train RL controller (default: 50000)
- `--seed`: Random seed (default: 42)
- `--output-dir`: Directory to save results (default: microgrid_system/results)

Example with custom settings:

```bash
python run_experiments.py --days 14 --battery-capacity 15 --train-rl --rl-timesteps 100000
```

## Output

The experiment produces the following outputs in the results directory:

1. CSV file with summary statistics for each controller
2. Plots comparing controllers based on various metrics
3. Time series plots showing the behavior of each controller
4. Saved RL model (if trained)

## Extending the Project

To extend this project, you can:

1. Implement additional control strategies
2. Enhance the environment with more complex dynamics
3. Add more realistic data generation or use real-world data
4. Improve the RL algorithm or add more sophisticated ML approaches
5. Add additional components to the microgrid (e.g., wind generation, EV charging)

## License

This project is provided for educational purposes. 