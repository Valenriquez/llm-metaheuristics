 # Name: PSO_Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh
import benchmark_func as bf
import metaheuristic as mh
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
    ( # Search operator 1
    'swarm_dynamic',
    { 
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
    },
    'metropolis'
    ),
    (  
    'spiral_dynamic',
    {
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    },
    'greedy'
)
]

problem_id= 2  #cambiar segun el problema
instance = 1
dimension = 5
num_agents = 100
num_iterations = 400
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# The code defines a metaheuristic named PSO_Metaheuristic using the swarm_dynamic operator for exploration and spiral_dynamic for exploitation. 
# Parameters are set according to typical settings for these operators in PSO algorithms. 
# The selector 'metropolis' is chosen for both operators, which suggests that a probabilistic approach will be used where solutions are accepted based on a probability criterion.
# This setup aims to balance between global exploration and local exploitation, suitable for optimizing complex functions as demonstrated by the benchmark_func library in this script.