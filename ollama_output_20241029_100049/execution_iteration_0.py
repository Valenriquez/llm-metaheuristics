import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import numpy as np
import metaheuristic as mh
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
    ("swarm_dynamic", {"factor": 0.7, "self_conf": 2.54, "swarm_conf": 2.56, "version": "inertial", "distribution": "levy"}, "probabilistic"),
    ("random_sample", {}, "greedy"),
]

problem_id= 2
instance = 1
dimension = 2
num_agents = 100
num_iterations = 400
num_replicas = 30

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y


# Short explanation and justification:

# The swarm metaheuristic is chosen as the metaheuristic operator due to its ability to explore and exploit the solution space effectively.
# The levy distribution is selected as the distribution for the swarm dynamic operator, as it has been shown to be effective in optimizing problems with complex decision spaces.
# The probabilistic selector is chosen as it encourages agents to explore diverse solutions and exploit promising ones.
# The greedy selector is chosen as it encourages agents to make the best local decision based on their current state.