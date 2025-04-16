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
problem_id = 20    
instance = 1
dimension = 5     
num_agents= 85    
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
        ('random_search',
         {
             'scale': trial.suggest_float('scale', 0.01, 0.2),
             'distribution': trial.suggest_categorical('distribution', ['gaussian'])
         },
         'metropolis'),

        ('central_force_dynamic',
         {
             'gravity': trial.suggest_float('gravity', 0.001, 0.01),
             'alpha': trial.suggest_float('alpha', 0.01, 0.1),
             'beta': trial.suggest_float('beta', 1.5, 4.5),
             'dt': trial.suggest_float('dt', 1.0, 2.0)
         },
         'probabilistic'),

        ('firefly_dynamic',
         {
             'distribution': trial.suggest_categorical('distribution', ['gaussian']),
             'alpha': trial.suggest_float('alpha', 1.0, 2.0),
             'beta': trial.suggest_float('beta', 1.0, 2.0),
             'gamma': trial.suggest_float('gamma', 10.0, 150.0)
         },
         'all'),

        ('swarm_dynamic',
         {
             'factor': trial.suggest_float('factor', 0.7, 0.9),
             'self_conf': trial.suggest_float('self_conf', 2.0, 3.0),
             'swarm_conf': trial.suggest_float('swarm_conf', 2.0, 3.0),
             'version': trial.suggest_categorical('version', ['inertial', 'constriction']),
             'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
         },
         'greedy')
    ]

    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=85, num_iterations=100, num_replicas=30)

    return performance

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=100)  

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")