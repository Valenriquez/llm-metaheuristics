# Name: Random Search
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
        'local_random_walk',
        { 'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform' },
        'greedy'
    )
]

problem_id = 2 #cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# Local random walk is a basic, unbiased and simple optimization technique, it might be suitable for small-scale optimization problems. The probability of changing the position was set to 0.75 to balance exploration and exploitation.
# The random walk algorithm with a scale of 1.0 was chosen as the search operator because it provides a uniform probability distribution for all movements, which can be beneficial for simple optimization problems.
# The greedy selector was used as it provides a good balance between exploration and exploitation for small-scale optimization problems.