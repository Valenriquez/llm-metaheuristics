import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from P1 import P1

# Parámetros iniciales
problem_id = 1    
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
        ('random_search', {'scale': trial.suggest_float('scale', 0.01, 0.2), 'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])}, 'metropolis'),
        ('central_force_dynamic', {'gravity': trial.suggest_float('gravity', 0.01, 0.2), 'alpha': trial.suggest_float('alpha', 0.1, 0.3), 'beta': trial.suggest_float('beta', 1.2, 1.5), 'dt': trial.suggest_float('dt', 0.5, 0.7)}, 'probabilistic'),
        ('differential_mutation', {'expression': trial.suggest_categorical('expression', ['rand', 'best', 'current', 'current-to-best', 'rand-to-best', 'rand-to-best-and-current']), 'num_rands': trial.suggest_int('num_rands', 2, 4), 'factor': trial.suggest_float('factor', 1.0, 1.5)}, 'greedy'),
        ('firefly_dynamic', {'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy']), 'alpha': trial.suggest_float('alpha', 0.3, 0.6), 'beta': trial.suggest_float('beta', 0.7, 1.0), 'gamma': trial.suggest_int('gamma', 45, 70)}, 'all')
    ]

    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=num_agents, num_iterations=num_iterations, num_replicas=num_replicas)
    
    return performance

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=num_replicas) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)