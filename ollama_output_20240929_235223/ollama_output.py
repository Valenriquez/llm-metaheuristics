 # Name: MyCustomMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_crossover',
    {
        'type': 'two',
        'mating_pool_factor': 0.4
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve the Rastrigin function using a genetic algorithm with two-point crossover, which is specified by 'type': 'two'. This approach aims to explore the search space more effectively by combining genes from different parts of the population. The mating pool factor is set to 0.4, influencing how many individuals are selected for breeding in each generation. A metropolis selector is employed to determine the acceptance probability for new solutions during optimization.
