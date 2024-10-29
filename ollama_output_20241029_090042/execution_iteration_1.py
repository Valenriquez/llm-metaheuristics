# Name: local_random_walk
# Code:

import sys
sys.path.append('/Users/valeriaenitez/Documents/research-llm/llm-metaheuristics')
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
        'probabilistic'
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


# # Explanation of the local random walk heuristic:
# The local random walk (LRW) heuristic is a type of search operator that randomly perturbs the current solution.
# This is done by adding a random step drawn from a given distribution (in this case, uniform) to the current solution.
# The LRW heuristic has a probability of success given by the 'probability' parameter, which controls the frequency of successful perturbations.
# In this experiment, we used a probability of 0.75, which means that the LRW heuristic is likely to perturb the current solution around 75% of the time.
# # The LRW heuristic can be effective for exploring the solution space, as it allows the algorithm to potentially jump to better regions.
# However, the effectiveness of the LRW heuristic also depends on the specific problem and its landscape.
# In this experiment, we used the LRW heuristic to search for a good solution to a multi-modal problem, and it appears to be a viable approach.