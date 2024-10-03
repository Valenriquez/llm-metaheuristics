 # Name: AdaptiveRandomSearch
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'genetic_crossover',
        {
            'probability': 0.8,
            'mutation_rate': 0.1,
            'population_size': 50
        },
        'greedy'
    ),
    ( # Search operator 2
        'genetic_mutation',
        {
            'probability': 0.7,
            'mutation_rate': 0.2,
            'population_size': 40
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The AdaptiveRandomSearch metaheuristic is designed to handle ill-structured global optimization problems by employing a combination of genetic crossover and mutation operators. These operators are selected based on their ability to explore diverse solutions within the problem space. 
# Genetic crossover promotes diversity in the population by exchanging segments of candidate solutions, while genetic mutation introduces small random changes to the population members, aiding in exploration of new solution regions.
# The 'greedy' and 'metropolis' selectors are chosen for their effectiveness in navigating through the search space based on the principle of maximizing expected improvement. This approach helps in efficiently converging towards better solutions during the optimization process. 
# Parameter values were selected to balance exploration (mutation rate and population size) with exploitation (crossover probability). The 'greedy' selector is used for faster convergence, while 'metropolis' allows for more thorough search through potential solution landscapes.