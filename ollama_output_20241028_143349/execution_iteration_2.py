# Name: Random Search Operator

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
    (
        "random_search",
        {
            "distribution": "uniform",
            "scale": 1.0
        },
        "probabilistic"
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
# This operator randomly generates new candidate solutions according to the uniform distribution.
# It's a simple and fast way to explore the solution space, which can be useful for quickly estimating the optimality of a solution.
# This approach is justified as a simple baseline to compare with more complex and time-consuming optimization methods.
# In addition, it is worth noting that the performance of this random search depends on the scale of the search space and the quality of the fitness function.
# However, it can be used as a baseline for simple problems where a good solution can be quickly found through exhaustive search.