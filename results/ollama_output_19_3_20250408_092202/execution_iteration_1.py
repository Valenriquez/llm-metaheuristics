# This is the Python Iteration: 1
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
    (  # Search Operator 1
        'random_search',
        {
            'scale': 0.010800760667082718,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (  # Search Operator 2
        'central_force_dynamic',
        {
            'gravity': 0.008897746068093574,
            'alpha': 0.031353305920269865,
            'beta': 1.0061371465741544,
            'dt': 0.4321184723208964
        },
        'probabilistic'
    ),
    (  # Search Operator 3
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 1,
            'factor': 0.6103772925689208
        },
        'greedy'
    ),
    (  # Search Operator 4
        'gravitational_search',
        {
            'gravity': 0.5,
            'alpha': 0.03
        },
        'all'
    )
]    

problem_id = 19    
instance = 1
dimension = 3     
num_agents= 57    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
print("Métrica de rendimiento (median + IQR):", performance_metric)
print("Mejor posición encontrada:", best_position)