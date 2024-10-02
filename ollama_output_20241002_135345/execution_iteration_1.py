 # Name: CustomMetaheuristic
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
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named CustomMetaheuristic as it combines elements from various heuristics to create a unique approach for optimization problems. 
# We use the Rastrigin function due to its properties that are typical of many ill-structured global optimization problems, which makes it suitable for testing random search algorithms.
# The search operator 'swarm_dynamic' is chosen with parameters set to optimize performance as per standard settings in the literature. These include factor (influence on swarm behavior), self_conf (self-confidence parameter), and swarm_conf (swarm confidence parameter). The version 'inertial' suggests that inertia weights are dynamically adjusted, which can be beneficial for exploring diverse regions of the search space. Distribution is set to 'uniform', as it provides a balanced exploration and exploitation characteristic needed in many optimization scenarios.
# The selector 'metropolis' is used, which implements probabilistic selection based on Metropolis criteria, ensuring a balance between exploitation and exploration during the optimization process. This choice aligns with the guidance that random search algorithms are useful for ill-structured global optimization problems, where the objective function may be nonconvex, nondifferentiable, and possibly discontinuous over a continuous, discrete, or mixed continuous-discrete domain.
# The settings adhere to the constraints provided in parameters_to_take.txt, ensuring only valid operators and parameters are utilized.