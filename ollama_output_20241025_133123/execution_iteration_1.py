# Name: Random Walk Metaheuristic
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
    ( # Search operator 1: Local Random Walk
        "local_random_walk",
        {
            "probability": 0.75,
            "scale": 1.0,
            "distribution": "levy"
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
# The local random walk metaheuristic uses the local random walk search operator, which is effective for searching large solution spaces.
# This metaheuristic is a probabilistic variant that balances exploration and exploitation.
# It has a 75% probability of making a new random move and a 25% chance of accepting the current solution.
# The levy distribution is used to introduce a degree of uncertainty in the random moves, which helps the search process escape local optima.
# Overall, this metaheuristic is well-suited for complex optimization problems where a good balance between exploration and exploitation is necessary.