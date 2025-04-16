# This is the Python Iteration: 3
# Author: Your Name
# Date: Insert Date

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
    ('random_search',
     {
         'scale': 0.10698506550580128,
         'distribution': 'uniform'
     },
     'metropolis'),

    ('central_force_dynamic',
     {
         'gravity': 0.018352209628422718,
         'alpha': 0.02663166241351011,
         'beta': 2.1594491504387783,
         'dt': 1.6731609893158907
     },
     'probabilistic'),

    ('firefly_dynamic',
     {
         'distribution': 'uniform', # Updated distribution to match the parameters
         'alpha': 1.0,
         'beta': 2.1594491504387783, # Updated beta value from the parameters
         'gamma': 338.4168792309532
     },
     'greedy'),

    ('spiral_dynamic',
     {
         'radius': 0.8224734700752977,
         'angle': 23.623723555826622,
         'sigma': 0.0484545356000584
     },
     'all')
]    

problem_id = 12    
instance = 1
dimension = 3     
num_agents= 57    
num_iterations = 100 # Assuming a reasonable number of iterations for the spiral_dynamic heuristic
num_replicas = 10    # Number of replicas to run in parallel

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

print("Performance Metric:", performance_metric)
print("Best Position:", best_position)