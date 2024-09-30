 # Name: Metaheuristic with Custom Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)  # Example function, replace with your chosen benchmark function
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'genetic_crossover',
    {
        'pairing': 'random',
        'crossover': 'uniform',
        'mating_pool_factor': 0.4
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed to use a genetic crossover operator for population diversity and improvement. The 'genetic_crossover' operator with parameters set to random pairing, uniform crossover method, and mating pool factor of 0.4 is chosen. It operates within the probabilistic selector context, aiming to leverage stochastic processes for better exploration in search spaces. This approach ensures a balance between exploitation (using genetic operators) and exploration (via probabilistic selection), which is crucial for handling complex optimization tasks effectively. The use of diverse crossover methods allows adaptation to different problem structures, while the probabilistic selector promotes adaptive behavior based on uncertain conditions encountered during the search process.