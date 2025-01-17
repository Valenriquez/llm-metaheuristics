# Name: Adaptive Hybrid Swarm Optimization (AHSO)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem
prob = fun.get_formatted_problem()

# Define the hybrid operators and selectors based on feedback
heur = [
    (  # Search operator 1 - Swarm Dynamic with Constriction Version
        'swarm_dynamic',
        {
            'factor': 0.6050548259236057,
            'self_conf': 2.8117372741130886,
            'swarm_conf': 2.4103131131767412,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        # Search operator 2 - Local Random Walk with Gaussian Distribution
        'local_random_walk',
        {
            'probability': 0.75728359975165,
            'scale': 0.5395666541913919,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Adaptive Hybrid Swarm Optimization (AHSO) is a novel approach that combines the strengths of swarm dynamic optimization with local search techniques. The hybrid operator set includes both global search methods to explore the solution space and local search methods to refine the solutions, leveraging the Metropolis selector for global exploration and the greedy selector for rapid convergence in promising regions. This combination aims to balance exploration and exploitation, leading to more efficient and effective optimization of complex problems like Rastrigin's function.