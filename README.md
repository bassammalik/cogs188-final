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

The project compares five different control strategies:

1. **Rule-Based Controller**: A simple heuristic controller that follows predefined rules based on current conditions (solar, load, price, battery state of charge).

2. **Reinforcement Learning Controller (PPO)**: Uses Proximal Policy Optimization (PPO) to learn an optimal control policy through interaction with the environment.

3. **Forecast-Based Controller**: Uses machine learning to forecast future solar, load, and price values, then makes decisions based on these forecasts.

4. **Q-Learning Controller (Temporal Difference)**: Implements a tabular Q-learning approach that discretizes the state space and uses temporal difference learning to update state-action values after each step.

5. **Monte Carlo Controller**: Uses first-visit Monte Carlo methods to learn state-action values by collecting complete episodes and updating values based on observed returns.

## Project Structure

```
microgrid_system/
├── __init__.py
├── environment.py              # Main environment simulation
├── controllers/                # Control strategies
│   ├── __init__.py
│   ├── rule_based.py           # Rule-based controller
│   ├── rl_controller.py        # PPO reinforcement learning controller
│   ├── forecast_controller.py  # Forecast-based controller
│   ├── q_learning_controller.py # Q-learning controller
│   └── monte_carlo_controller.py # Monte Carlo controller
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
- `--train-q`: Train the Q-learning controller (default: False)
- `--q-episodes`: Number of episodes to train Q-learning controller (default: 200)
- `--train-mc`: Train the Monte Carlo controller (default: False)
- `--mc-episodes`: Number of episodes to train Monte Carlo controller (default: 200)
- `--seed`: Random seed (default: 42)
- `--output-dir`: Directory to save results (default: microgrid_system/results)
- `--enable-degradation`: Enable battery degradation modeling (default: False)
- `--enable-weather-uncertainty`: Enable weather uncertainty with cloud events (default: False)
- `--plot-degradation`: Generate and save battery degradation plots (default: False)
- `--plot-weather`: Generate and save weather event impact plots (default: False)

Example with custom settings:

```bash
python run_experiments.py --days 14 --battery-capacity 15 --train-rl --train-q --train-mc --enable-degradation
```

## Output

The experiment produces the following outputs in the results directory:

1. CSV file with summary statistics for each controller
2. Plots comparing controllers based on various metrics
3. Time series plots showing the behavior of each controller
4. Saved models for RL, Q-learning, and Monte Carlo controllers (if trained)
5. Training progress plots for Q-learning and Monte Carlo controllers
6. Battery degradation and weather uncertainty plots (if enabled)

## Reinforcement Learning Approaches

The project implements three different reinforcement learning approaches:

1. **PPO (Proximal Policy Optimization)**: A policy gradient method that uses a neural network to approximate the policy and value functions. It's implemented using Stable-Baselines3.

2. **Q-Learning (Temporal Difference)**: A value-based method that learns the action-value function (Q-function) by bootstrapping from the next state's value. It uses a tabular representation with state space discretization.

3. **Monte Carlo Control**: A value-based method that learns from complete episodes. It updates the action-value function based on the actual returns observed after visiting a state-action pair.

These approaches represent different paradigms in reinforcement learning:
- PPO: Deep RL with function approximation
- Q-Learning: Tabular RL with bootstrapping (TD learning)
- Monte Carlo: Tabular RL with episode-based learning (no bootstrapping)

## Environment Features

The environment includes optional features to increase realism:

1. **Battery Degradation**: Models capacity loss due to cycling and calendar aging
2. **Weather Uncertainty**: Simulates cloud events and forecast errors

