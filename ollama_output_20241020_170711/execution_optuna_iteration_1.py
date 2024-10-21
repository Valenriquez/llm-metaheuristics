import optuna
import sys

sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# Function to evaluate the performance of a sequence of operators
def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    num_cores = multiprocessing.cpu_count()
    results_parallel = Parallel(n_jobs=num_cores)(delayed(run_metaheuristic)() for _ in range(num_replicas))

    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

# Define the objective function for Optuna optimization
def objective(trial):
    # Generate operators based on trial parameters
    heur = [
        # Random flight operator with levy distribution
        {'operator': 'random_flight', 'scale': trial.suggest_float('random_flight_scale', 0.1, 0.9), 'distribution': 'levy', 'beta': trial.suggest_float('random_flight_beta', 0.1, 0.9)},

        # Local random walk operator with uniform distribution
        {'operator': 'local_random_walk', 'probability': trial.suggest_float('local_random_walk_probability', 0.1, 0.9), 'scale': trial.suggest_float('local_random_walk_scale', 0.1, 0.9), 'distribution': 'uniform'},

        # Spiral dynamic operator
        {'operator': 'spiral_dynamic', 'radius': trial.suggest_float('spiral_dynamic_radius', 0.1, 0.9), 'angle': trial.suggest_float('spiral_dynamic_angle', 0.1, 0.9), 'sigma': trial.suggest_float('spiral_dynamic_sigma', 0.1, 0.9)},

        # Swarm dynamic operator with inertial version and Gaussian distribution
        {'operator': 'swarm_dynamic', 'factor': trial.suggest_float('swarm_dynamic_factor', 0.1, 0.9), 'self_conf': trial.suggest_float('swarm_dynamic_self_conf', 0.1, 0.9), 'swarm_conf': trial.suggest_float('swarm_dynamic_swarm_conf', 0.1, 0.9), 'version': 'inertial', 'distribution': 'gaussian'}
    ]

    # Select the benchmark function
    fun = bf.Rastrigin(2)

    # Create the problem
    prob = fun.get_formatted_problem()

    # Evaluate the performance of the sequence
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

# Create an Optuna study
study = optuna.create_study(direction="minimize")

# Optimize the objective function
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and performance
print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)