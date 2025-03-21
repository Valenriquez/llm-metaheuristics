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
    ('random_search', 
     {
         'scale': 0.01, 
         'distribution': 'gaussian'
     }, 
     'metropolis'),
    ('central_force_dynamic', 
     {
         'gravity': 0.005, 
         'alpha': 0.02, 
         'beta': 1.0, 
         'dt': 0.5
     }, 
     'probabilistic'),
    ('differential_mutation', 
     {
         'expression': 'rand-to-best-and-current', 
         'num_rands': 2, 
         'factor': 1.5
     }, 
     'greedy'),
    ('gravitational_search', 
     {
         'gravity': 0.5, 
         'alpha': 0.03
     }, 
     'all')
]    

# New parameters to be used in the search operators
new_params = {
    'scale': 0.005035628717552679,
    'distribution': 'levy',
    'gravity': 0.003590119618730883,
    'alpha': 0.026832619108314713,
    'beta': 0.8478804245390078,
    'dt': 0.38849502519862067,
    'expression': 'rand-to-best-and-current',
    'num_rands': 4,
    'factor': 1.2097488679790935
}

# Update the heur list with the new parameters
heur[0] = ('random_search', {'scale': new_params['scale'], 'distribution': new_params['distribution']}, 'metropolis')
heur[1] = ('central_force_dynamic', {'gravity': new_params['gravity'], 'alpha': new_params['alpha'], 'beta': new_params['beta'], 'dt': new_params['dt']}, 'probabilistic')
heur[2] = ('differential_mutation', {'expression': new_params['expression'], 'num_rands': new_params['num_rands'], 'factor': new_params['factor']}, 'greedy')
heur[3] = ('gravitational_search', {'gravity': new_params['gravity'], 'alpha': new_params['alpha']}, 'all')

problem_id = 1  # Example problem ID
instance = 1   # Example instance
dimension = 10  # Example dimension
num_agents = 50  # Example number of agents
num_iterations = 1000  # Example number of iterations
num_replicas = 10  # Example number of replicas

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

print(f"Performance Metric: {performance_metric}")
print(f"Best Position: {best_position}")
print("Fitness Array:")
print(fitness_array)