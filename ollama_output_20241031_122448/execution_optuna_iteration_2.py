# Name: Swarm Metaheuristic with Inertial Version and Gaussian Distribution Enhanced with Optuna

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

# Note: If a word is in the code do not remove it, but if a number is in the code, replace it with "trial.suggest_float('variable_name', 0.1, 0.9)"
def objective(trial):
    factor = trial.suggest_float('factor', 0.1, 0.9)
    self_conf = trial.suggest_float('self_conf', 0.1, 0.9)
    swarm_conf = trial.suggest_float('swarm_conf', 0.1, 0.9)

    heur = [
        (  # Search operator 1
            'swarm_dynamic',
            {
                'factor': factor,
                'self_conf': self_conf,
                'swarm_conf': swarm_conf,
                'version': 'inertial',
                'distribution': 'gaussian'
            },
            'probabilistic'
        )
    ]

    fun = bf.Rastrigin(2) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

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