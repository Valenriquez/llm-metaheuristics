# This is the Python Iteration: 4
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
        'central_force_dynamic',
        {
            'gravity': 0.00885796398485526,
            'alpha': 0.013394178279628039,
            'beta': 3.2183504359562596,
            'dt': 0.8283869172978505
        },
        'metropolis'
    ),
    (  # Search Operator 2
        'genetic_crossover',
        {
            'pairing': 'rank',
            'crossover': 'blend',
            'mating_pool_factor': 0.6499011589805156
        },
        'probabilistic'
    ),
    (  # Search Operator 3
        'random_sample',
        {},
        'all'
    ),
    (  # Search Operator 4
        'swarm_dynamic',
        {
            'factor': 1.0428215658613365,
            'self_conf': 2.312214331464137,
            'swarm_conf': 2.5115896725687583,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'uniform'
    )
]

# Rest of the code remains unchanged