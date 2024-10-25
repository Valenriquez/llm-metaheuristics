 # Name: Metaheuristic Exploration with Greedy Selection
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
        'sample',
        { 
            'parameter1': value1,
            'parameter2': value2,
            ... more parameters as needed
        },
        'greedy'
    ),
    (  
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7 or 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial" or "constriction",
            'distribution': "uniform" or "gaussian" or "levy"
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
# This code defines a sequence of metaheuristic operators to be used in an optimization task. The operators include 'sample', 'spiral_dynamic', and 'swarm_dynamic' with specific parameters for each, all set to use the 'greedy' selector. These operators are applied iteratively on a benchmark problem defined by its ID, instance, dimension, number of agents, iterations, and replicas. The goal is to explore solutions using these metaheuristic techniques to find the optimal fitness as defined by the benchmark problem.
# Each operator has been chosen based on their ability to handle different types of optimization problems and their parameters have been set according to typical values recommended for such tasks in literature. The 'greedy' selector ensures that each step is evaluated with a bias towards immediate improvement, which can be efficient but may not explore the entire solution space thoroughly. This approach is justified by the need for quick convergence in some cases where finding an acceptable solution quickly outweighs the benefits of exhaustive search.