# Name: Hybrid Metaheuristic with Random Search and Genetic Operators

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1 - Random Search with Gaussian distribution
        'random_search',
        {
            'scale': 0.005286234693202038,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'swarm_dynamic',  # Swarm dynamic with Inertial version
        {
            'factor': 0.6985102963231126,
            'self_conf': 2.136374742160269,
            'swarm_conf': 2.691754642797297,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',  # Spiral dynamic with given parameters
        {
            'radius': 0.7034564041054041,
            'angle': 22.094605228683537,
            'sigma': 0.06277387239751933
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines the Random Search operator with a Gaussian distribution to explore the solution space efficiently. 
# The Swarm dynamic operator is used with an Inertial version, leveraging the social behavior of particles to guide solutions towards optima.
# Additionally, Spiral dynamic is included for its ability to exploit promising regions of the search space while avoiding premature convergence.