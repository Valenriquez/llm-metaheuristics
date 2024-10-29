# Name: Dynamic Metaheuristic

# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh
from P1 import P1

def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    met = mh.Metaheuristic(prob, heur, num_agents=num_agents, num_iterations=num_iterations)
    met.verbose = True
    met.run()
    best_position, f_best = met.get_solution()
    return f_best, best_position

heur = [
    (
        "dynamic",
        {
            "radius": 0.9,
            "angle": 22.5,
            "sigma": 0.1
        },
        "metropolis"
    ),
    (
        "swarm_dynamic",
        {
            "factor": 0.7,
            "self_conf": 2.54,
            "swarm_conf": 2.56,
            "version": "inertial",
            "distribution": "uniform"
        },
        "all"
    )
]

problem_id=2  # cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 100
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# The "dynamic" operator combines exploration and exploitation to find the optimal solution in the Rastrigin function.
# The "metropolis" selector ensures that the algorithm moves in the direction of better solutions.
# The "swarm_dynamic" operator allows for adaptivity in the algorithm by using self-confidence and swarm-confidence parameters.
# The "inertial" version ensures that the algorithm remembers the previous movements and takes into account the direction of better solutions.
# The "uniform" distribution helps to explore the search space uniformly, reducing the chance of getting stuck in local optima.
# The use of two operators with different properties can help to avoid being stuck in local optima.