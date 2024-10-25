# Name: Levy-Flight

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
    ),
    (
    "swarm_dynamic",
    {
        "factor": 0.7,
        "self_conf": 2.54,
        "swarm_conf": 2.56,
        "version": "constriction",
        "distribution": "levy"
    },
    "greedy"
    )
]

problem_id= 2 # cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# This Levy-Flight based metaheuristic is chosen because it combines the advantages of both global search capabilities of Levy-Flight and the exploration-exploitation balance of swarm optimization.
# The parameters are chosen to balance exploration and exploitation in the swarm optimization algorithm, while the levy flight provides the necessary diversity for the solution space. 
# This combination can help in effectively exploring the search space and reaching the optimal solution.