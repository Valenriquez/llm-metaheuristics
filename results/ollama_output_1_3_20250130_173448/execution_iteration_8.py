# This is the Python Iteration: 9
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
    ("central_force_dynamic", 
     {
         "gravity": 0.0020600266064054167, 
         "alpha": 0.019405335623674164, 
         "beta": 1.8118530203705738, 
         "dt": 0.7787400213036669
     }, 
     "metropolis"
    ),
    ("swarm_dynamic", 
     {
         "factor": 0.5699960334410429,
         "self_conf": 2.6855136020040655, 
         "swarm_conf": 2.5378481104573525,
         "version": "constriction",
         "distribution": "uniform"
     }, 
     "greedy"
    ),
    ("spiral_dynamic", 
     {
         "radius": 0.8027061128454661, 
         "angle": 24.8541624959335, 
         "sigma": 0.13046695703312078
     }, 
     "gaussian"
    )
]

problem_id = 1    
instance = 1
dimension = 3     
num_agents= 57    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

print(f"Performance Metric: {performance_metric}")
print(f"Best Position: {best_position}")