import numpy as np
import ioh
import sys
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh
from P1 import P1

def evaluate_sequence_IOH(sequence, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()

    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.verbose = True
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
    #final_fitness = np.array([x[-1] for x in fitness_array.T])
    print("final_fitness_array", fitness_array)

    # Retorna el mejor valor y la mejor posición encontrada en todas las réplicas
    best_fitness_index = np.argmin(fitness_values)
    best_position = positions[best_fitness_index]
    
    return performance_metric, best_position

# Parámetros de entrada
sequence = [
    (  # Search operator 1
        '[operator_name]',
        {
            'parameter1': value1,
            'parameter2': value2,
            more parameters as needed
        },
        '[selector_name]'
    ),
    (
        '[operator_name]',
        {
            'parameter1': value1,
            'parameter2': value2,
            ... more parameters as needed
        },
        '[selector_name]'
    )
]

problem_id = {self.problem_id}  
instance = 1
dimension = {self.dimensions}
num_agents = {self.num_of_agents}
num_iterations = 100
num_replicas = 30  # Ajusta el número de réplicas según sea necesario

# Ejecución de la función con réplicas
performance_metric, best_position = evaluate_sequence_IOH(sequence, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
#print("Métrica de rendimiento (Mediana + IQR):", performance_metric)
#print("Mejor posición encontrada:", best_position)

# Obtener y comparar con el óptimo
problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y
#print("Fitness óptimo:", optimal_fitness)
