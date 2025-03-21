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
problem_id = 14    
instance = 1
dimension = 5     
num_agents = 85    
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
        ("random_search",
         {
             "scale": trial.suggest_float('scale', 0.001, 0.02),
             "distribution": trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
         },
         "metropolis"),
        ("central_force_dynamic",
         {
             "gravity": trial.suggest_float('gravity', 0.001, 0.01),
             "alpha": trial.suggest_float('alpha', 0.01, 0.05),
             "beta": trial.suggest_float('beta', 1.0, 2.0),
             "dt": trial.suggest_float('dt', 1.0, 2.0)
         },
         "probabilistic"),
        ("differential_mutation",
         {
             "expression": trial.suggest_categorical('expression', ['rand', 'best', 'current', 'current-to-best', 'rand-to-best', 'rand-to-best-and-current']),
             "num_rands": 2,
             "factor": trial.suggest_float('factor', 0.6, 1.0)
         },
         "greedy"),
        ("genetic_crossover",
         {
             "pairing": trial.suggest_categorical('pairing', ['rank', 'cost', 'random', 'tournament_2_100']),
             "crossover": trial.suggest_categorical('crossover', ['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5']),
             "mating_pool_factor": trial.suggest_float('mating_pool_factor', 0.3, 0.7)
         },
         "all")
    ]
    
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=num_agents, num_iterations=num_iterations, num_replicas=num_replicas)
    
    return performance

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=num_replicas)

# Final performance evaluation
best_heur = study.best_params['heur']
ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
fun = P1(variable_num=dimension, problem=ioh_problem)
prob = fun.get_formatted_problem()
performance = evaluate_sequence_performance(best_heur, prob, num_agents=num_agents, num_iterations=num_iterations, num_replicas=30)

print(f"Best performance: {performance}")