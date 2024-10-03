 # Name: Custom Genetic Algorithm
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
            'elitism': True,
            'mutation_rate': 0.01
        },
        'greedy'
    ),
    ( # Search operator 2
        'genetic_mutation',
        {
            'probability': 0.3,
            'mutation_rate': 0.05
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Custom Genetic Algorithm" is designed to solve optimization problems using a combination of genetic crossover and mutation operators. 
# The genetic_crossover operator uses a probability of 0.8 for exchanging genetic material between solutions, with elitism enabled to preserve the best individuals in each generation. 
# It also includes a mutation rate of 0.01 to introduce random changes, promoting diversity within the population.
# The genetic_mutation operator operates by changing individual genes in the population with a lower probability (0.3) but higher mutation rate (0.05), which helps in exploring different areas of the solution space.
# Both operators are controlled by a greedy selector that decides whether to accept an improved solution, making it suitable for local search. The metropolis selector is used for probabilistic acceptance based on energy differences, useful for global exploration. 
# These settings aim to balance convergence speed with the ability to escape local minima through diverse mutations and crossovers.