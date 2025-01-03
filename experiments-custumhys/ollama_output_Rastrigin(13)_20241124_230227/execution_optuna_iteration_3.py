# Optuna Algorithm for Metaheuristic

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
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
        # Search operator 1
('swarm_dynamic',
{
    'factor': trial.suggest_float('swarm_factor', 0.5, 1.0),
    'self_conf': trial.suggest_float('self_confidence', 2.0, 3.0),
    'swarm_conf': trial.suggest_float('swarm_confidence', 2.0, 3.0),
    'version': trial.suggest_categorical('swarm_version', ['inertial', 'constriction']),
    'distribution': trial.suggest_categorical('swarm_distribution', ['uniform', 'gaussian', 'levy'])
},
'random_walk'
),
(
'spiral_dynamic',
{
    'radius': trial.suggest_float('spiral_radius', 0.5, 0.9),
    'angle': trial.suggest_float('spiral_angle', 15.0, 30.0),
    'sigma': trial.suggest_float('spiral_sigma', 0.05, 0.2)
},
'acceptance_probability'
)
    ]

    fun = bf.Rastrigin(13)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)