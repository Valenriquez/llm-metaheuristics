```python
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
from P1 import P1

# Parámetros iniciales
problem_id = 5    
instance = 1
dimension = 6     
num_agents = 98    
num_iterations = 100
num_replicas = 10

# Definir la función para evaluar el rendimiento de la secuencia
def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    # Número de núcleos disponibles para paralelización
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores, prefer="threads")(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Calcular la métrica de rendimiento usando la mediana y el IQR
    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

# Definir el objetivo de Optuna
def objective(trial):
    heur = [
        ('random_search', {'scale': trial.suggest_float('scale', 0.005, 0.02), 'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])}, 'greedy'),
        ('central_force_dynamic', {'gravity': trial.suggest_float('gravity', 0.0005, 0.0015), 'alpha': trial.suggest_float('alpha', 0.005, 0.015), 'beta': trial.suggest_float('beta', 1.2, 1.8), 'dt': trial.suggest_float('dt', 0.5, 1.5)}, 'probabilistic'),
        ('differential_mutation', {'expression': trial.suggest_categorical('expression', ['rand', 'best', 'current']), 'num_rands': trial.suggest_int('num_rands', 1, 4), 'factor': trial.suggest_float('factor', 0.5, 2.0)}, 'metropolis'),
        ('firefly_dynamic', {'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy']), 'alpha': trial.suggest_float('alpha', 0.8, 1.2), 'beta': trial.suggest_float('beta', 0.7, 0.9), 'gamma': trial.suggest_int('gamma', 60, 180)}, 'all')
    ]

    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=num_agents, num_iterations=num_iterations, num_replicas=num_replicas)
    
    return performance

# Crear el study y ejecutar la búsqueda de hiperparámetros
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Imprimir los mejores parámetros y valor objetivo
print("Mejores parámetros encontrados:")
print(study.best_params)
print("Mejor valor objetivo encontrado:")
print(study.best_value)
```