# Name: SwarmDynamic-Rastrigin

# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh
from P1 import P1

heur = [
    (
        "swarm_dynamic",
        {
            "radius": 0.9,
            "angle": 22.5,
            "sigma": 0.1
        },
        "metropolis"
    )
]

problem_id = 2  #cambiar segun el problema
instance = 1
dimension = 2
num_agents = 100
num_iterations = 100
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y


# Short explanation and justification:
# The Swarm Dynamic operator is chosen as it is known for its robustness and adaptability. The metropolis selector is used to adapt the exploration-exploitation trade-off. The radius and angle parameters are set to provide a good balance between exploration and exploitation. The sigma parameter is set to provide some noise in the algorithm. The algorithm is expected to efficiently search the solution space of the Rastrigin function.