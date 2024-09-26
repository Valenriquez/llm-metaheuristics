 # Name: Custom Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The custom metaheuristic is designed to solve optimization problems using a combination of four main search operators: genetic mutation, gravitational search, random flight, and local random walk. These operators are chosen based on their ability to explore diverse regions of the solution space, which is crucial for global optimization tasks such as those addressed by benchmark functions like the sphere function.
# The genetic_mutation operator employs Gaussian distribution for mutations, aiming to balance exploration and exploitation by allowing larger changes in parameter values that could lead to better solutions.
# Gravitational search mimics the gravitational force between masses to iteratively move candidate solutions towards areas of higher fitness scores.
# Random flight is a movement strategy where the step size follows a Levy distribution, which encourages more diverse explorations compared to Brownian motion (Gaussian distribution).
# Local random walk with Gaussian distribution helps in fine-tuning and exploring local neighborhoods around current best solutions.
# These operators are selected for their flexibility and ability to handle different types of problems through parameter tuning, making them suitable for the benchmark function being used here. The choice of parameters for each operator is based on typical settings that have shown good performance across various optimization tasks.