# Name: Random Levy
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
    ("random_flight", 
     {
        "scale": 1.0,
        "distribution": "levy",
        "beta": 1.5
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

# Explanation:
# This metaheuristic combines the Levy flight operator with the greedy selection.
# The random_flight operator helps the search to spread and diversify in the solution space.
# The greedy selection then helps to select the best moves in this expanded space.
# By using the Levy distribution, we can simulate a search process that exhibits characteristic power-law features, potentially helping the algorithm to explore more effectively.