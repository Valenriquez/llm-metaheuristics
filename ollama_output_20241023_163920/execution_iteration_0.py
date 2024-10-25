 # Name: GeneticAlgorithmWithCustomOperators
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
    'spiral_dynamic',
    { 
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    },
    'metropolis'
    ),
    (  
    'swarm_dynamic',
    {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
    },
    'probabilistic'
)]

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
# The code defines a genetic algorithm with two custom operators: 'spiral_dynamic' and 'swarm_dynamic'. 
# The 'spiral_dynamic' operator uses parameters radius (0.9), angle (22.5), and sigma (0.1) to guide the search in a spiral pattern, while the 'swarm_dynamic' operator employs factor (0.7), self_conf (2.54), swarm_conf (2.56), version ('inertial'), and distribution type ('uniform') to simulate swarm behavior. 
# Both operators use 'metropolis' as their selector for probabilistic selection, which is appropriate given the nature of these operators aiming for dynamic exploration in the search space.