import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))

import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from P1 import P1

# Parámetros iniciales
problem_id = 1    
instance = 1
dimension = 3     
num_agents = 57    
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
        ('central_force_dynamic',
         {
             'gravity': trial.suggest_float('gravity', 0.001, 0.01),
             'alpha': trial.suggest_float('alpha', 0.02, 0.03),
             'beta': trial.suggest_float('beta', 1.45, 1.5),
             'dt': trial.suggest_float('dt', 0.75, 0.78)
         },
         'greedy'),
        ('firefly_dynamic',
         {
             'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy']),
             'alpha': trial.suggest_float('alpha', 1.1, 1.2),
             'beta': trial.suggest_float('beta', 1.1, 1.15),
             'gamma': trial.suggest_float('gamma', 90, 100)
         },
         'greedy'),
        ('spiral_dynamic',
         {
             'radius': trial.suggest_float('radius', 0.8, 0.9),
             'angle': trial.suggest_float('angle', 22, 25),
             'sigma': trial.suggest_float('sigma', 0.1, 0.12)
         },
         'greedy')
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