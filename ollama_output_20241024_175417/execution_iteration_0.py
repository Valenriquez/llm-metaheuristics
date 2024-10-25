
# Name: Levy Walk Metaheuristic
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
        "local_random_walk",
        {
            "probability": 0.75,
            "scale": 1.0,
            "distribution": "levy"
        },
        "greedy"
    ),
]

problem_id=2
instance = 1
dimension = 5
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# The Levy Walk Metaheuristic is a type of random walk algorithm that uses the Levy distribution to generate new solutions. This metaheuristic is particularly useful for solving optimization problems with noisy or non-differentiable objective functions.
# The local_random_walk operator is used as the search operator, which allows the algorithm to explore different regions of the solution space.
# The greedy selector is used to select the best solution from each iteration, which helps the algorithm converge to a good solution quickly.