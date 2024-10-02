 # Name: Basic Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'add__operator__',
    {
        'parameters': ""
    },
    'add__selector__'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a basic metaheuristic for optimizing the Rastrigin function with two dimensions using a simple search operator and selector as specified. The Rastrigin function is chosen because it is a common benchmark for testing optimization algorithms, particularly those based on random or genetic methods due to its multimodal nature.
# The search operator 'add__operator__' is used without any specific parameters, which might limit the effectiveness of the search. It also uses 'add__selector__', ensuring that if a genetic-based method is selected (which typically requires both crossover and mutation), then it will use them accordingly as per the guidelines from parameters_to_take.txt. This setup allows for basic exploration of the function's space, though more sophisticated settings or additional operators might yield better results depending on the specific characteristics of the optimization problem.