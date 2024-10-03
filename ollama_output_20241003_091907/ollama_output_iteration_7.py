 # Name: Adaptive Random Search with Greedy Selection
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'gaussian',
        {
            'scale': 0.5,
            'distribution': 'normal'
        },
        'greedy'
    ),
    ( # Search operator 2
        'levy',
        {
            'scale': 0.7,
            'distribution': 'levy'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic Adaptive Random Search with Greedy Selection is designed to optimize the Sphere function. Two search operators are employed: Gaussian for exploration and Levy for exploitation. Gaussian uses a normal distribution with scale 0.5, while Levy uses a levy distribution with scale 0.7. The greedy selector ensures that only the best candidates are retained, promoting convergence towards an optimal solution. This approach aims to balance exploration of diverse areas in the search space with exploitation of potentially promising regions.
# Each operator is paired with specific parameters for tuning their behavior: Gaussian's normal distribution and Levy's levy distribution, which affect how the algorithm explores and exploits different parts of the problem space. The greedy selector ensures that only the most effective candidates are retained, guiding the search towards better solutions more efficiently. This combination helps in effectively navigating complex non-convex landscapes typical in ill-structured global optimization problems.