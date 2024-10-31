# Name: Enhanced Genetic Algorithm with Optuna Optimization
# Code:

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import  population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# Write the whole function
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

# Note: If a word is in the code do not remove it, but if a number is in the code, replace it with "trial.suggest_float('variable_name', 0.1, 0.9)"
def objective(trial):
    heur = [
        trial.suggest_float('ga_prob_crossover', 0.1, 0.9),
        trial.suggest_int('ga_num_parents_mating', 2, 7),
        trial.suggest_float('ga_tournament_selection_prob', 0.2, 0.9),
        trial.suggest_int('ga_population_size', 100, 200),
        trial.suggest_int('ga_num_generations', 100, 300),
        trial.suggest_float('ga_mutation_probability', 0.1, 0.9),
        trial.suggest_float('ga_elitism_rate', 0.1, 0.9),
        trial.suggest_int('ga_num_threads', 1, multiprocessing.cpu_count()),
    ]

    fun = bf.Rastrigin(2)  # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

# Write the whole code
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)