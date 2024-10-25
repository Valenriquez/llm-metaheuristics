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
# The random flight metaheuristic is a variant of the Levy flight algorithm, which uses a levy distribution to model the movement of particles. This algorithm has been shown to be effective in solving optimization problems with complex landscapes. In this implementation, we use a scale factor of 1.0 and a beta value of 1.5, which allows for a good balance between exploration and exploitation.
# The greedy selector is used to select the next solution based on its fitness value.
# This metaheuristic has been shown to be effective in solving optimization problems with complex landscapes and can be used as an alternative to other optimization algorithms such as genetic algorithms or simulated annealing.