# Name: Levy-Based Random Walk (LRW)
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
    "local_random_walk",
    {
        "probability": 0.75,
        "scale": 1.0,
        "distribution": "levy"
    },
    "probabilistic"
    ),
]

problem_id= 2 #cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# # This is a metaheuristic-based solution for the optimization problem. It uses the local_random_walk operator with probabilistic selection.
# # The "probability" parameter controls the probability of acceptance of a new solution, and the "scale" parameter controls the magnitude of the levy distribution.
# # The "distribution" parameter specifies the distribution to be used (in this case, levy), and the "probabilistic" selector indicates that the new solution is chosen randomly based on its fitness.