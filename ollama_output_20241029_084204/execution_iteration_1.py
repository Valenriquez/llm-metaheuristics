# Name: Local Levy Greedy Search
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
        {"probability": 0.75, "scale": 1.0, "distribution": "levy"},
        "greedy"
    ),
    (
        "spiral_dynamic",
        {"radius": 0.9, "angle": 22.5, "sigma": 0.1},
        "greedy"
    ),
    (
        "random_sample",
        {},
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
# This metaheuristic uses the combination of the Local Random Walk, Spiral Dynamic and Random Sample operators in a greedy manner to find the optimal solution.
# The Local Random Walk operator helps to explore the solution space, while the Spiral Dynamic operator provides guidance towards the optimal solution.
# The Random Sample operator helps to reduce the chance of getting stuck in local optima.
# The greedy selection method ensures that the operators with the highest fitness values are chosen.