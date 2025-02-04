# This is NOT the metaheuristic template
# This is the optuna template

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
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
        ('random_search', {'scale': trial.suggest_float('scale', 0.01, 0.9), 'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])}, 'metropolis'),
        ('genetic_crossover', {'pairing': trial.suggest_categorical('pairing', ['rank', 'cost', 'random', 'tournament_2_100']), 'crossover': trial.suggest_categorical('crossover', ['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5']), 'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)}, 'probabilistic'),
        ('genetic_mutation', {'scale': trial.suggest_float('scale', 1.0, 3.0), 'elite_rate': trial.suggest_float('elite_rate', 0.1, 0.5), 'mutation_rate': trial.suggest_float('mutation_rate', 0.1, 0.9), 'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])}, 'metropolis')
    ]

    fun = bf.Rastrigin(3)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=57, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)