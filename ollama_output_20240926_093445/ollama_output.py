 # Name: Metaheuristic Exploration
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
         'pairing': 'random',
         'crossover': 'uniform',
         'mating_pool_factor': 0.4
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation:
# The metaheuristic is named "Metaheuristic Exploration." It utilizes the Rastrigin function as a benchmark problem due to its multi-modal nature, which allows for exploration of multiple optimal solutions. 
# The genetic crossover operator with random pairing and uniform crossover method is chosen based on parameters_to_take.txt. This approach encourages diversity in the population while maintaining some level of similarity between parents to facilitate convergence towards better solutions.
# The heuristic includes a single search operator from the genetic operators, specifically 'genetic_crossover' with specified parameters for pairing and crossover type. 
# The selector used is 'greedy', which prioritizes solutions that improve upon the current best solution in each iteration. This approach aims to leverage quickly converging towards better fitness values while avoiding premature convergence by exploring multiple potential optimal points.
