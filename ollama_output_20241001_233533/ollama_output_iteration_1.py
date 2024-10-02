 # Name: Dynamic Swarm Metaheuristic
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
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named "Dynamic Swarm Metaheuristic" as it incorporates a swarm dynamic algorithm with specific parameters for exploration in the search space. 
# The Rastrigin function, which is a benchmark problem suitable for continuous optimization tasks, is selected to evaluate the performance of our metaheuristic. 
# A swarm_dynamic operator is chosen with carefully set parameters: 'factor' is set to 0.7, enhancing exploration by allowing particles to adjust their velocities more dynamically; 'self_conf' and 'swarm_conf' are both set to 2.54 and 2.56 respectively, influencing the cognitive and social components of each particle's movement; 'version' is set to 'inertial', which allows for a balance between exploration and exploitation by incorporating an inertial term in the velocity update equation; 'distribution' is set to 'uniform', ensuring that particles are initialized with velocities drawn from a uniform distribution, promoting diversity in initial search points.
# The selector used here is 'probabilistic', suitable for stochastic optimization problems where the probability of selecting certain actions or operators plays a crucial role in decision making. This choice allows for random exploration and adjustment of swarm dynamics based on probabilistic rules, which can be beneficial in complex and non-convex optimization landscapes.