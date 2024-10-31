"""
Using the num_replicas

"""
import optuna
import sys
import benchmark_func as bf
import matplotlib.pyplot as plt
import matplotlib as mpl
import population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import ioh
from P1 import P1

sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

# Parámetros iniciales
problem_id = 1  
instance = 1
dimension = 5
num_agents = 100
num_iterations = 400
num_replicas = 1

# Definir la función para evaluar el rendimiento de la secuencia
def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    # Número de núcleos disponibles para paralelización
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores)(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Calcular la métrica de rendimiento usando la mediana y el IQR
    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

# Definir el objetivo de Optuna
def objective(trial):
    heur = [
        ('spiral_dynamic',
         {'radius': trial.suggest_float('radius', 0.01, 0.9),
          'angle': trial.suggest_float('angle', 0.01, 22.5),
          'sigma': trial.suggest_float('sigma', 0.01, 0.1)},
         'metropolis'),
        ('swarm_dynamic',
         {'factor': trial.suggest_float('factor', 0.01, 0.7),
          'self_conf': trial.suggest_float('self_conf', 0.01, 2.54),
          'swarm_conf': trial.suggest_float('swarm_conf', 0.01, 2.56),
          'version': 'inertial',
          'distribution': 'uniform'},
         'probabilistic')
    ]

    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)
