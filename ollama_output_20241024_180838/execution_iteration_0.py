# Name: Metaheuristic_Rastrigin
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh


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
        "swarm_dynamic",
        {
            "factor": 0.7,
            "self_conf": 2.54,
            "swarm_conf": 2.56,
            "version": "inertial",
            "distribution": "uniform"
        },
        "greedy"
    )
]

problem_id=2
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y


# Short explanation and justification:
# This metaheuristic uses the swarm dynamic operator with a factor of 0.7 to balance exploration and exploitation.
# The self-confidence is set to 2.54 and swarm confidence to 2.56 to allow for some degree of adaptability.
# The inertial version is chosen as it tends to favor more exploratory moves.
# The uniform distribution allows for an even spread of the search space, which can help in finding the global optimum.
# This metaheuristic is expected to perform well on the Rastrigin function due to its ability to balance exploration and exploitation.