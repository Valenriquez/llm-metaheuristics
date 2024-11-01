# Name: Enhanced Optuna-Enhanced Metaheuristic for Rastrigin(2) Optimization

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
        (  # Search operator 1
        'rand_search',
        {
            'population_size': trial.suggest_int('population_size', 10, 100),
            'num_iterations': trial.suggest_int('num_iterations', 100, 1000),
            'elite_ratio': trial.suggest_float('elite_ratio', 0.1, 0.9)
        },
        'tournament'
    ),
    (  # Search operator 2
        'mutation',
        {
            'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.9),
            'mutation_strength': trial.suggest_float('mutation_strength', 0.01, 0.9)
        },
        'random'
    )
    ]
    fun = bf.Rastrigin(2)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)