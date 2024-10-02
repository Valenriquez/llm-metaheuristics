 # Name: Custom Genetic Algorithm with Swarm Dynamics
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_crossover',
    {
        'probability': [0.9],
        'angle': [22.5],
        'sigma': [0.1]
    },
    'greedy'
)] + [( # Search operator 2
    'swarm_dynamic',
    {
        'factor': [0.7, 1.0],
        'self_conf': [2.54],
        'swarm_conf': [2.56],
        'version': ['inertial', 'constriction'],
        'distribution': ['uniform', 'gaussian', 'levy']
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines a genetic algorithm with crossover for exploration and swarm dynamics optimization using the metropolis selector for exploitation. 
# The genetic crossover operator is configured with parameters probability, angle, and sigma to facilitate diverse search directions.
# Swarm dynamics are optimized using factors ranging from 0.7 to 1.0, self-confidence set at 2.54, swarm confidence at 2.56, and versions including inertial and constriction, exploring different distributions such as uniform, gaussian, and levy for a broader search space.
# The greedy selector is chosen for the genetic crossover to ensure that better solutions are favored during exploration, while the metropolis selector is used for the swarm dynamics to balance between exploitation and exploration based on probabilistic criteria.