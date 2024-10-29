# Name: Levy Search Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenquezlimon/Documents/research-llm/llm-metaheuristics')
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
    ('levy_search', { 'scale': 1.0, 'distribution': 'levy', 'beta': 1.5 }, 'probabilistic'),
]

problem_id=2
instance=1
dimension=2
num_agents=100
num_iterations=400
num_replicas=1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# # This is a Levy Search metaheuristic. 
# # It uses the Levy distribution to generate search steps. 
# # The scale parameter controls the step size. 
# # The distribution parameter determines the type of distribution used. 
# # The beta parameter controls the trade-off between exploration and exploitation. 
# # The probabilistic selector means that the next search step is chosen probabilistically based on the probability given by the heuristic function. 
# # This operator can be used for optimizing complex objective functions that have multiple local optima.