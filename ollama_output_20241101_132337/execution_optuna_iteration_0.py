# Name: Enhanced Metaheuristic using Crossover and Mutation Operators

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import optuna
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
        (  # Crossover operator
            'crossover',
            {
                'p': trial.suggest_float('p', 0.1, 0.9)  # Crossover probability
            },
            'tournament'  # Tournament selector
        ),
        (  # Mutation operator
            'mutation',
            {
                'p': trial.suggest_float('p', 0.1, 0.9),  # Mutation probability
                'sigma': trial.suggest_float('sigma', 0.1, 1.0)  # Mutation strength
            },
            'uniform'  # Uniform selector
        )
    ]
    fun = bf.Rastrigin(2)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)