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
        ('random_search', {
            'scale': trial.suggest_float('scale', 0.1, 2.0),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
        }, 'greedy'),
        ('central_force_dynamic', {
            'gravity': trial.suggest_float('gravity', 0.0001, 0.01),
            'alpha': trial.suggest_float('alpha', 0.001, 0.05),
            'beta': trial.suggest_float('beta', 1.0, 4.5),
            'dt': trial.suggest_float('dt', 0.5, 2.0)
        }, 'metropolis'),
        ('differential_mutation', {
            'expression': trial.suggest_categorical('expression', ['rand', 'best', 'current']),
            'num_rands': trial.suggest_int('num_rands', 1, 3),
            'factor': trial.suggest_float('factor', 0.5, 2.0)
        }, 'probabilistic'),
    ]

    fun = bf.Cigar(3) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=57, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)