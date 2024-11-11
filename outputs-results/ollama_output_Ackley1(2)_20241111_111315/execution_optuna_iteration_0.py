import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import matplotlib.pyplot as plt

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
            'random_flight',
            {
                'scale': trial.suggest_float('scale', 0.01, 0.5),
                'distribution': trial.suggest_categorical('distribution', ['gaussian']),
                'beta': trial.suggest_int('beta', 2, 4)
            },
            'greedy'
        ),
        (  # Search operator 2
            'genetic_mutation',
            {
                'scale': trial.suggest_float('scale', 0.01, 0.5),
                'elite_rate': trial.suggest_float('elite_rate', 0.01, 0.1),
                'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.25),
                'distribution': trial.suggest_categorical('distribution', ['gaussian']),
            },
            'greedy'
        ),
    ]
    
    fun = bf.Ackley1(2)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)