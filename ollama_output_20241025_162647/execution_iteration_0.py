
# Name: Spiral Dynamic

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
    "spiral_dynamic",
    { 
      "radius": 0.9,
      "angle": 22.5,
      "sigma": 0.1
    },
    "greedy"
    ),
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
# Spiral dynamic is a well-known optimization method, especially useful for functions that are noisy or multimodal. It iteratively moves in a spiral pattern, with parameters that influence its exploration and exploitation capabilities. Here, I'm using a greedy selection method to choose the next point to evaluate. This choice is based on the previous function evaluations, which will lead to a better optimization performance.

