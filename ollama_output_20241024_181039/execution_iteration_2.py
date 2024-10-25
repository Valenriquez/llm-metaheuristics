# Name: Random Flight Metaheuristic
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
        "random_flight",
        {
            "scale": 1.0,
            "distribution": "levy",
            "beta": 1.5
        },
        "greedy"
    )
]

problem_id=2 #cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y


# Short explanation and justification:
# The Random Flight metaheuristic is a probabilistic optimization algorithm that uses the flight of particles to search for the optimal solution. 
# It starts by initializing a set of particles with random positions and velocities, then iteratively updates their positions based on the Levy distribution.
# The scale parameter controls the magnitude of the jump, while the beta parameter determines the shape of the distribution.
# In this implementation, we use the greedy selection strategy to choose the best solution among the particles.
# This metaheuristic is suitable for problems with multiple local optima, as it can efficiently explore different regions of the search space.