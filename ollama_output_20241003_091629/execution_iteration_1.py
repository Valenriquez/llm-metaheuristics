 # Name: Custom Genetic Algorithm with Random Search
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1: Genetic Crossover
        'genetic_crossover',
        {
            'mutation_rate': 0.1,
            'population_size': 50,
            'crossover_rate': 0.8,
        },
        'greedy' # Selector name from parameters_to_take.txt
    ),
    ( # Search operator 2: Random Search with Mutation
        'genetic_mutation',
        {
            'mutation_rate': 0.1,
            'population_size': 50,
        },
        'all' # Selector name from parameters_to_take.txt
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines two main operators for exploration and exploitation: genetic crossover and random mutation. Genetic crossover allows for the exchange of genetic material between solutions, promoting diversity in the population. Random mutation introduces small variations into the solution space, aiding in exploration to avoid local minima. The 'greedy' selector prioritizes promising individuals by selecting them with a higher probability, while the 'all' selector considers all candidates equally, providing broader search capabilities. These operators and selectors are chosen based on their effectiveness for ill-structured global optimization problems as specified in parameters_to_take.txt.
