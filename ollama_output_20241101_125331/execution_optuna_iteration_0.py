# Name: Adaptive Metropolis-Greedy Algorithm with Optuna-Enhanced Parameters
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

# Note: If a word is in the code do not remove it, but if a number is in the code, replace it with "trial.suggest_float('variable_name', 0.1, 0.9)"
def objective(trial):
    heur = [
        (trial.suggest_categorical('operator_1', ['local_random_walk', 'greedy']),  # Replace with your chosen operator
        {
            'probability': trial.suggest_float('probability_1', 0.1, 0.9),  # Replace with your chosen parameters
            'scale': trial.suggest_float('scale_1', 0.1, 0.9),
            'distribution': trial.suggest_categorical('distribution_1', ['gaussian', 'uniform'])
        },
        trial.suggest_categorical('selector_1', ['metropolis', 'greedy'])  # Replace with your chosen selector
    ),
        (trial.suggest_categorical('operator_2', ['local_random_walk', 'greedy']),  # Replace with your chosen operator
        {
            'probability': trial.suggest_float('probability_2', 0.1, 0.9),  # Replace with your chosen parameters
            'scale': trial.suggest_float('scale_2', 0.1, 0.9),
            'distribution': trial.suggest_categorical('distribution_2', ['gaussian', 'uniform'])
        },
        trial.suggest_categorical('selector_2', ['metropolis', 'greedy'])  # Replace with your chosen selector
    )
]

    fun = bf.Rastrigin(2) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

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

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)