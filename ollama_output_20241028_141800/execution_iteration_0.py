# Name: Spiral Dynamic Metaheuristic
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
    'spiral_dynamic',
    {
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    },
    'probabilistic'
    ),
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

# # Spiral dynamic metaheuristic is inspired by the behavior of certain animals, which have a natural ability to move in spiral shapes while foraging.
# # This operator mimics this behavior, by perturbing the current position by a random amount, ensuring the exploration of the solution space.
# # The radius and angle parameters allow for control over the spread and direction of the perturbation, while the sigma parameter controls the magnitude of the random displacement.