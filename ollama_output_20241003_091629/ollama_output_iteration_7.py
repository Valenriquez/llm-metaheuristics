 # Name: RandomSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search Operator 1
        'genetic_crossover',
        {
            'probability': 0.8,
            'mutation_rate': 0.1,
            'population_size': 50
        },
        'greedy'
    ),
    ( # Search Operator 2
        'genetic_mutation',
        {
            'mutation_rate': 0.1,
            'probability': 0.8,
            'population_size': 50
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed as a Random Search Metaheuristic using two operators: genetic crossover and genetic mutation. The genetic crossover operator involves exchanging parts of the solution between individuals to create new solutions, while the genetic mutation operator introduces random changes in the solution to explore different regions of the search space. The selectors used are greedy for one operator and metropolis for the other, allowing both exploration and exploitation during the optimization process. Parameters such as probability, mutation rate, and population size are set according to typical values recommended for these operators in a random search approach. This combination aims to efficiently navigate through the problem landscape by leveraging both explorative and exploitative strategies.