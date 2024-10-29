# Name: Optuna-Enhanced Metaheuristic for Solving Multi-Dimensional Function Optimization
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
from P1 import P1

problem_id = trial.suggest_categorical("problem_id", [1, 2])
instance = trial.suggest_categorical("instance", [1, 2, 3, 4, 5])
dimension = trial.suggest_categorical("dimension", [2, 5, 10, 20])
num_agents = trial.suggest_categorical("num_agents", [5, 10, 15, 20, 25, 50])
num_iterations = trial.suggest_categorical("num_iterations", [10, 50, 100, 200, 400])
num_replicas = trial.suggest_categorical("num_replicas", [1, 5, 10, 15, 20])

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
        (
    'local_search',
    {
        "scale": trial.suggest_float("scale", 0.1, 0.9),
        "distribution": trial.suggest_categorical("distribution", ["uniform", "gaussian", "uniform"]),
        "beta": trial.suggest_float("beta", 1.0, 3.0)
    },
    "greedy"
    ),
    (
    'local_random_walk',
    {
        "probability": trial.suggest_float("probability", 0.1, 0.9),
        "scale": trial.suggest_float("scale", 0.1, 0.9),
        "distribution": trial.suggest_categorical("distribution", ["uniform", "gaussian"])
    },
    "greedy"
    )
    ]

                
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)