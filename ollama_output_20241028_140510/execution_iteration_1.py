# Name: local_search_metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh


heur = [
    (
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'probabilistic'
    ),
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
# This metaheuristic is designed to locally explore the search space with a probability-driven operator, 'local_random_walk'. The chosen operator, local_random_walk, is a probabilistic operator that changes the current position by a random amount within a given scale. The distribution is set to uniform to allow for exploration in all directions. This metaheuristic aims to quickly converge to the global optimum.