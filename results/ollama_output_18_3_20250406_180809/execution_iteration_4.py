
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
    (  # Swarm Optimization with Constriction Version
        'swarm',
        {
            'self_conf': 2.85,
            'swarm_conf': 2.60,
            'version': 'constriction',
            'scale': 0.0548 * 1.1,
            'gravity': 0.0707 + 0.05,
            'alpha': 0.0624 * 1.1,
            'beta': 1.7978 - 0.1,
            'dt': 0.2525 * 1.1,
            'expression': 'gaussian',
            'factor': 0.7243 + 0.05
        },
        'tournament'
    ),
    (  # Velocity Update with Adaptive Components
        'velocity_update',
        {
            'self_conf': 2.85,
            'swarm_conf': 2.60,
            'version': 'constriction',
            'scale': 0.0548 * 1.1,
            'gravity': 0.0707 + 0.05,
            'alpha': 0.0624 * 1.1,
            'beta': 1.7978 - 0.1,
            'dt': 0.2525 * 1.1,
            'expression': 'gaussian',
            'factor': 0.7243 + 0.05
        },
        'tournament'
    ),
    (  # Particle Update with Adaptive Parameters
        'particle_update',
        {
            'self_conf': 2.85,
            'swarm_conf': 2.60,
            'version': 'constriction',
            'scale': 0.0548 * 1.1,
            'gravity': 0.0707 + 0.05,
            'alpha': 0.0624 * 1.1,
            'beta': 1.7978 - 0.1,
            'dt': 0.2525 * 1.1,
            'expression': 'gaussian',
            'factor': 0.7243 + 0.05
        },
        'tournament'
    ),
    (  # Selection Operator with Tournament Selection
        'selection',
        {
            'num_parents': int(len(population) / 2)
        },
        'tournament'
    )
]    

problem_id = 18    
instance = 1
dimension = 3     
num_agents= 57    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
print("Métrica de rendimiento (Mediana + IQR):", performance_metric)
print("Mejor posición encontrada:", best_position)

# Obtener y comparar con el óptimo
problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y                 
            