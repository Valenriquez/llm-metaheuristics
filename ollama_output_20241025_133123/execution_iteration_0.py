# Name: Random Search with Local Random Walk and Random Sampling
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
    ( # Search operator 1
        "random_flight",
        {
            "scale": 1.0,
            "distribution": "levy",
            "beta": 1.5
        },
        "greedy"
    ),
    (  
        "local_random_walk",
        {
            "probability": 0.75,
            "scale": 1.0,
            "distribution": "uniform"
        },
        "greedy"
    ),
    (  
        "random_sample",
        {},
        "greedy"
    )
]

problem_id=2 # cambiar según el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y


# Short explanation and justification:
# The proposed metaheuristic, "Random Search with Local Random Walk and Random Sampling", combines the principles of three search operators to solve the problem.
# "Random Flight" is used to jump to a new location with a scale of 1.0 and a Levy distribution with a beta of 1.5.
# "Local Random Walk" is used to make a move in the direction of the gradient with a probability of 0.75, scale of 1.0, and a uniform distribution.
# "Random Sample" is used to select a new solution from the population.
# This combination allows the algorithm to jump between different areas of the search space and refine its search within a given area.
# The greedy selection method is used to choose the best solution in each iteration.
# This algorithm is particularly effective for problems where the search space is highly complex and contains multiple local optima.