 # Name: Dynamic Swarm Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the dynamic swarm operator with parameters from parameters_to_take.txt
heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'operator': 'swarm_dynamic',
        'parameters': {
            'factor': 0.7,  # Using a value that is typical for swarm dynamics
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',  # Choosing an inertial version of the operator
            'distribution': 'uniform'  # Uniform distribution in the swarm movement
        },
        'selector': 'greedy'  # Using a greedy selector to direct the search
    }
)]

# Create and run the metaheuristic with the defined operators
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Dynamic Swarm Metaheuristic" is designed to optimize the Rastrigin function, which is a standard benchmark for testing optimization algorithms due to its multimodal nature. The algorithm utilizes a dynamic swarm approach where particles adjust their movement based on local and global best solutions with parameters influenced by typical values observed in similar studies. 
# The core operator 'swarm_dynamic' employs an inertial version that allows for both self-confidence ('self_conf') and swarm-wide confidence ('swarm_conf'). These are set according to standard settings, ensuring a balanced exploration and exploitation trade-off. The movement distribution is uniform across the swarm, which aids in exploring the search space evenly. 
# A greedy selector is chosen to guide the search towards better solutions, leveraging local information for efficient convergence. This setup should provide a balance between exploration and exploitation typical of dynamic swarming behaviors found in nature, expected to perform well on problems like the Rastrigin function where many minima exist within its multi-modal landscape.