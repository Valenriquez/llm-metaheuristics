 # Name: Custom Swarm Dynamic with Probabilistic Selection
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
        'factor': [0.7, 1.0],
        'self_conf': [2.54],
        'swarm_conf': [2.56],
        'version': ['inertial', 'constriction'],
        'distribution': ['uniform', 'gaussian', 'levy']
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is a custom swarm dynamic algorithm with probabilistic selection. This approach is suitable for global optimization problems like the Rastrigin function, which has multiple local minima. 
# The parameters are set to explore different factors of influence on the swarm dynamics, self confidence, swarm confidence, and distribution methods. By using probabilistic selection, the algorithm focuses its search efforts more effectively in promising areas of the solution space. This method is particularly useful for problems with nonconvex landscapes where a diverse set of exploration techniques can lead to better convergence properties.
# The use of different factors (0.7 to 1.0) allows for flexibility in adjusting the influence of local and global information, while self_conf and swarm_conf are tuned to balance between individual and collective intelligence. The inclusion of multiple distributions (uniform, gaussian, levy) ensures a broad exploration across the search space, reducing the risk of getting trapped in局部最优解.