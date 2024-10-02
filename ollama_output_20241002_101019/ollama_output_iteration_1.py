 # Name: PSOInspiredMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# PSO (Particle Swarm Optimization) is a population-based algorithm inspired by the social behavior of bird flocking or fish schooling. It operates on a set of candidate solutions, called particles, which 'fly' through the problem space by following the current best known positions and velocities.
heur = [( # Search operator 1
    'GeneticCrossover',
    {
        'crossover_rate': 0.9,
        'mutation_rate': 0.1,
        # ... more parameters as needed
    },
    'GeneticMutation'
)]

# The GeneticCrossover operator will be used to create new solutions by combining parts of existing solutions, while the GeneticMutation operator will introduce small random changes in the population to explore new areas of the search space.
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# # PSOInspiredMetaheuristic is designed to mimic the behavior of particle swarm optimization (PSO), a popular metaheuristic for solving global optimization problems. The algorithm uses genetic operators such as crossover and mutation to iteratively improve candidate solutions. In this implementation, we use GeneticCrossover with a high crossover rate and a lower mutation rate to encourage exploration while ensuring that new solutions are derived from existing ones. This approach is suitable for continuous optimization problems like the Rastrigin function due to its ability to balance between exploitation of current best solutions and exploration of new areas in the search space. The parameters used (crossover_rate, mutation_rate) are typical for PSO and genetic algorithms, aligning with the guidelines provided in parameters_to_take.txt.
