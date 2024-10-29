# Name: Levy Metropolis
# Code:

import sys
sys.path.append('/Users/valeriaenrico limon/Documents/research-llm/llm-metaheuristics')
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
        "levy",
        {"scale": 1.0, "distribution": "levy", "beta": 1.5},
        "metropolis"
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


# # Short explanation and justification:
# The levy metropolis metaheuristic uses the levy distribution to guide the search process, which is known for its robustness and efficiency.
# The metropolis selector is used to decide whether to accept or reject the new solution based on its quality.
# This metaheuristic is well-suited for optimization problems with complex and rugged landscapes.