# Optuna-Enhanced Metaheuristic

# Code:

import optuna
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import  population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

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

def objective(trial):
    heur = [
        trial.suggest_float('mutation_rate', 0.1, 0.9),
        trial.suggest_int('population_size', 10, 50),
        trial.suggest_float('crossover_rate', 0.5, 0.9),
        # ... add other hyperparameters as needed
    ]

    fun = bf.Rastrigin(2)  # Example problem
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

# Configure and run Optuna study
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

# Print results
print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)

# Save hyperparameters to file
with open('parameters_to_take.txt', 'w') as f:
    f.write(str(study.best_params))
