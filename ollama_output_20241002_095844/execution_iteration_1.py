 # Name: PSOwithCustomOperator
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh


fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'PSOOperator',
    {
        'parameter1': 0.7,
        'parameter2': 0.9,
        # ... more parameters as needed
    },
    'PSOSelector'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Short explanation and justification:
# The metaheuristic chosen is named PSOwithCustomOperator. This name reflects the integration of a custom operator (inspired by Particle Swarm Optimization) with standard selector methods, in this case, PSOSelector for both genetic crossover and mutation based on the given parameters. 
# The benchmark function used here is Rastrigin(2), which is a suitable choice for testing optimization algorithms due to its multiple local minima, making it an ill-structured global optimization problem suitable for random search algorithms.
# PSOOperator is defined with two main parameters: parameter1 and parameter2. These are set to 0.7 and 0.9 respectively, based on typical settings in PSO literature for balancing exploration and exploitation. The first parameter influences the personal best influence, while the second one affects the global best influence within the swarm dynamics.
# PSOSelector is chosen as both a crossover and mutation method are used together in Particle Swarm Optimization to explore solutions across the search space effectively. This choice aligns with the requirement from parameters_to_take.txt that if genetic_crossover is selected, then genetic_mutation must also be included.
# The metaheuristic runs for 100 iterations as specified, and results are printed in a formatted output showing the best solution found (x_best) and its corresponding fitness value (f_best). This setup allows for experimentation and comparison with other optimization techniques to evaluate performance on complex non-convex functions like Rastrigin.