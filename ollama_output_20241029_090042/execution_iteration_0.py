# Name: local_search_with_operators
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
    'local_search',
    {
        "scale": 1.0,
        "distribution": "uniform",
        "beta": 1.5
    },
    "greedy"
    ),
    (
    'local_random_walk',
    {
        "probability": 0.75,
        "scale": 1.0,
        "distribution": "uniform"
    },
    "greedy"
    )
]

problem_id = 2
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# Local search is an optimization technique that is used to find the minimum or maximum of a function.
# The two operators, 'local_search' and 'local_random_walk', are both forms of local search.
# The 'local_search' operator moves to a better neighbor with probability "scale".
# The 'local_random_walk' operator randomly walks through the neighborhood.
# Both operators have a "selector" that chooses the next move.
# By combining these two operators with the greedy selection strategy, we create a more robust local search method.
# This combination of operators can lead to faster convergence to the optimum.
# However, the performance of this method highly depends on the specific problem and its structure.
# Further experiments are needed to determine the effectiveness of this combination for different problems.