# Name: Enhanced Genetic Algorithm
# Code:

import population as pp
import operators as op
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the metaheuristic using the given operators and selectors
heur = [
    (op.GeneticMutation, {'crossover_rate': 0.1}),
    (op.GeneticCrossover, {'version': 'blend'}),
    (op.GreedySelector, {}),
]

# Create the metaheuristic object
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic
met.run()

# Print the best solution found
print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Explanation:
# The Enhanced Genetic Algorithm uses the GeneticMutation operator to modify the population, 
# while the GeneticCrossover operator is used to combine individuals from the population. 
# The GreedySelector operator selects the best individual from the population for reproduction.

# Benefits:
# - Genetic operators are well-suited for optimization problems with complex decision variables.
# - Genetic algorithms are known for their ability to find good solutions in a wide variety of problems.
# - The Enhanced Genetic Algorithm combines genetic operators with a greedy selection strategy, 
#   which can improve the performance of the algorithm in comparison to other metaheuristic algorithms.