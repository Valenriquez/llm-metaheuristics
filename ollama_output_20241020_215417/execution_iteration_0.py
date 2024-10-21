 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

gravitational_search_params = {
    "gravity": 1.0,
    "alpha": 0.02
}

heur = [
    (
        "gravitational_search",
        gravitational_search_params,
        "all"
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search operator with specified parameters for gravity and alpha. The selector is set to "all" which means all operators will be used in each iteration of the algorithm. The Rastrigin function is chosen as the benchmark problem, and it has 2 dimensions. The metaheuristic runs for 100 iterations, and its results are printed including the best solution found (x_best) and its corresponding fitness value (f_best).
# The gravitational search operator is used with specified parameters to control the behavior of the search. Gravity represents the strength of the gravitational force, influencing how strongly agents are attracted to better solutions. Alpha controls the scaling factor for movement based on the gradient direction towards the best solution found so far. This setup allows the algorithm to balance between exploration and exploitation during its optimization process.