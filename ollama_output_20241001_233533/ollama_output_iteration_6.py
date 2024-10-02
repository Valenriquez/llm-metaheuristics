 # Name: Adaptive Search Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'parameters': {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        }
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Adaptive Search Metaheuristic is designed to dynamically adapt its search strategy based on the problem's characteristics, utilizing a swarm dynamics approach with parameters that encourage exploration and exploitation balance. The 'swarm_dynamic' operator employs an inertial version with Gaussian distribution for particle movement, allowing it to efficiently navigate through the solution space. The 'metropolis' selector ensures that the best-performing solutions are retained, promoting convergence towards optimal regions while avoiding premature convergence. This approach is suitable for continuous optimization problems where adaptability and exploration are crucial.