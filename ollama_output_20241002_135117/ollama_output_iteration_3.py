 # Name: Adaptive Random Search with Genetic Crossover
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Using Inertial Swarm with Genetic Crossover and Metropolis Selection
heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': "inertial",
        'distribution': "uniform"
    },
    'metropolis'
)]

# Using Genetic Crossover and Mutation for Evolutionary Search
heur += [( # Search operator 2
    'genetic_crossover',
    {
        'mutation_rate': 0.1,
        'population_size': 50
    },
    'metropolis'
)]

# Using Genetic Mutation for Evolutionary Search
heur += [( # Search operator 3
    'genetic_mutation',
    {
        'mutation_rate': 0.1,
        'population_size': 50
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is designed to adapt between random search (inertial swarm with uniform distribution) and genetic algorithms for optimization problems, particularly those modeled by the Rastrigin function which is suitable for continuous, non-differentiable, and possibly discontinuous domains. 
# A key aspect of this approach is the use of both genetic crossover and mutation to explore and refine the solution space, guided by the metropolis selector that decides between accepting new solutions based on probability, enhancing exploration without being trapped in local minima. The inertial swarm dynamic allows for a balance between global and local search, ensuring broad coverage while fine-tuning through adaptive parameters.
# The use of genetic operators is justified due to their effectiveness in handling complex non-convex landscapes, while the metropolis selector balances between exploitation and exploration by allowing probabilistic acceptance of worse solutions for diversity.