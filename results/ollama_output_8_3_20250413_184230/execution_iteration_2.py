# This is the Python Iteration: 2
# Author: [Your Name]
# Date: [Insert Date]

# Code:
import sys
from pathlib import Path
import ioh
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from P1 import P1

def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
        
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()

    def run_metaheuristic():
        met = mh.Metaheuristic(prob, heur, num_agents, num_iterations)
        met.verbose = False
        met.run()
        best_position, f_best = met.get_solution()
        return f_best, best_position

    # Ejecutar en paralelo el número de réplicas
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores, prefer="threads")(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Extraer los valores de fitness de los resultados y calcular la métrica de rendimiento
    fitness_values = [result[0] for result in results_parallel]
    positions = [result[1] for result in results_parallel]
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    # Fitness finales
    fitness_array = np.array(fitness_values).T
    print("final_fitness_array", fitness_array)

    # Retorna el mejor valor y la mejor posición encontrada en todas las réplicas
    best_fitness_index = np.argmin(fitness_values)
    best_position = positions[best_fitness_index]
    return performance_metric, best_position, fitness_array

heur = [
    ("random_search",
     {
         "scale": 0.022769852469246794,
         "distribution": "uniform"
     },
     "probabilistic"),
    ("central_force_dynamic",
     {
         "gravity": 0.19569591881288007,
         "alpha": 0.29834693396304784,
         "beta": 2.6111058069773674,
         "dt": 0.8792213496925175
     },
     "greedy"),
    ("differential_mutation",
     {
         "expression": "current",
         "num_rands": 2,
         "factor": 1.0021238191094728
     },
     "metropolis"),
    ("spiral_dynamic",
     {
         "radius": 0.8228336709854503,
         "angle": 23.799376040542374,
         "sigma": 0.11841577235320079
     },
     "all")
]    

problem_id = 8    
instance = 1
dimension = 3     
num_agents= 57    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
print("Performance Metric:", performance_metric)
print("Best Position:", best_position)