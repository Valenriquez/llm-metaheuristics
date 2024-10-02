 # Name: Custom Genetic Algorithm with Modified Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_mutation',
    {
        'scale': 0.5,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'uniform'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic is named "Custom Genetic Algorithm with Modified Operators." This name reflects the use of a genetic mutation operator tailored for our specific needs, as outlined in parameters_to_take.txt.
# A Rastrigin function is chosen due to its multimodal nature, which is suitable for testing optimization algorithms. The problem is formatted into a standard benchmark format using `get_formatted_problem()`.
# We define the first search operator as 'genetic_mutation'. Parameters are set as follows:
  # 'scale': 0.5 - This parameter influences the magnitude of mutations in the genetic algorithm, allowing for broader or more focused exploration of the solution space.
  # 'elite_rate': 0.1 - Defines the proportion of the population considered elites and directly carried over to the next generation, promoting exploitation of promising solutions.
  # 'mutation_rate': 0.25 - Controls the probability that a mutation occurs during reproduction, balancing exploration and exploitation in the search process.
  # 'distribution': 'uniform' - Specifies that mutations should follow a uniform distribution, which can help in exploring different parts of the solution space uniformly.
# The selector used is 'probabilistic', meaning that based on predefined probabilities, this operator will be applied during each iteration to potentially improve the population.
# The metaheuristic runs for 100 iterations, with detailed output enabled (`verbose = True`). After execution, the best solution found is printed along with its fitness value.