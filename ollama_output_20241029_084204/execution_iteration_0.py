# Name: Levy Flight Based Metaheuristic for IOH Problem
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
    {
      "probability": 0.75,
      "scale": 1.0,
      "distribution": "levy"
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
# This metaheuristic is designed to solve IOH problem using local_random_walk operator, which is suitable for exploring complex search spaces. 
# The levy distribution is used to simulate Levy flights, which is expected to help the algorithm escape local optima. 
# The greedy selector is used to ensure the selection of the most promising candidate solution. 
# This metaheuristic can potentially solve the IOH problem more efficiently and effectively by utilizing the unique properties of Levy flights.