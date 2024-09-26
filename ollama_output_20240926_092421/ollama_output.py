 # Name: MyCustomMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)  # Example function, change to your desired benchmark function
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'genetic_crossover',
    {
        'pairing': 'rank',
        'crossover': 'blend',
        'mating_pool_factor': 0.4
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# This code defines a custom metaheuristic named MyCustomMetaheuristic, which uses the Rastrigin function as its benchmark problem. The heuristic employs genetic crossover with specific parameters including 'pairing' set to 'rank', 'crossover' to 'blend', and 'mating_pool_factor' at 0.4. It utilizes all four selector methods: greedy, metropolis, probabilistic, and random sampling. T
# he metaheuristic runs for 100 iterations with verbose output enabled, printing the best solution found after optimization.
