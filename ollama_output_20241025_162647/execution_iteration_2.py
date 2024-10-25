# Name: Levy-Flight-Metropolis-Merge

# Code:

import sys
sys.path.append('/Users/valeriaenquezlimon/Documents/research-llm/llm-metaheuristics')
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
    "metropolis"
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
    "all"
    ),
    (
    "spiral_dynamic",
    {
        "radius": 0.9,
        "angle": 22.5,
        "sigma": 0.1
    },
    "probabilistic"
    ),
    (
    "swarm_dynamic",
    {
        "factor": 0.7,
        "self_conf": 2.54,
        "swarm_conf": 2.56,
        "version": "inertial",
        "distribution": "gaussian"
    },
    "metropolis"
    )
]

problem_id= 2 #cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# The proposed metaheuristic combines different search operators, each with its unique characteristics, to improve the exploration and exploitation of the search space. The 'random_flight' operator provides the initial population with Levy-like distribution, 'local_random_walk' offers localized search and exploration, 'random_sample' ensures the sampling of the search space, 'spiral_dynamic' contributes with the dynamic adjustments of the parameters, and 'swarm_dynamic' enables the swarm behavior to the metaheuristic. This combination is justified as it allows the metaheuristic to explore different regions of the search space, making it more effective in finding the optimal solution.