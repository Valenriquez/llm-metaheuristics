# Name: local_random_walk
# Code:

import sys
sys.path.append('/Users/valeriaenitezlimon/Documents/research-llm/llm-metaheuristics')
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
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'greedy'
    )
]

problem_id= 2 
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# This local_random_walk heuristic uses the probability parameter to control the exploration-exploitation trade-off. The uniform distribution allows for random jumps in the search space, which can be beneficial for complex problems. The greedy selector is used to decide when to explore or exploit the current best solution. This metaheuristic can be effective for solving problems with diverse solution landscapes.