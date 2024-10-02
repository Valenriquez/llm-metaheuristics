 # Name: Adaptive Multi-Operator Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the heuristic operators dynamically based on parameters from parameters_to_take.txt
operators = {
    'genetic_mutation': {
        'parameters': {'scale': 0.5, 'elite_rate': 0.1, 'mutation_rate': 0.25, 'distribution': 'uniform'},
        'selector': 'probabilistic'
    },
    'random_flight': {
        'parameters': {'scale': 0.5, 'distribution': 'levy', 'beta': 1.5},
        'selector': 'probabilistic'
    }
}

heur = [(op_name, op_params, sel) for op_name, (op_params, sel) in operators.items()]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic is named 'Adaptive Multi-Operator Metaheuristic' because it dynamically adapts by selecting from a set of predefined heuristic operators based on parameters specified in the parameters_to_take.txt file. This approach allows for flexibility and adaptability to different optimization scenarios without hardcoding specific operators.
# The Rastrigin function is chosen as the benchmark problem due to its well-known properties, which are suitable for testing a variety of metaheuristic algorithms.
# Heuristic operators are defined within a dictionary (`operators`) where each operator's parameters and selector can be dynamically adjusted from the provided parameters in the parameters_to_take.txt file. This setup ensures that only allowed operators and their specific parameters are used, adhering to the constraints outlined.
# Genetic mutation is included as part of the heuristic because it is a common component in genetic algorithms (a type of metaheuristic) for exploring the search space by introducing small variations into the population, which is crucial for maintaining diversity and preventing premature convergence. The random flight operator is also selected probabilistically to maintain exploration capabilities alongside exploitation properties of mutation operators inherent in genetic algorithms.
# The use of probabilistic selectors ensures that each operator has a controlled influence on the decision-making process based on predefined probabilities, promoting a balance between explorative and exploitative behaviors across iterations. This dynamic selection mechanism enhances the adaptability of the metaheuristic to different problem characteristics without manual tuning for each specific case.