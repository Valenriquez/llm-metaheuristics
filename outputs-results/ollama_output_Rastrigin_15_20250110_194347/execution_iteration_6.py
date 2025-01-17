# Name: Enhanced Hybrid Metaheuristic (EHM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {
            'population_size': 100,
            'mutation_rate': 0.05
        },
        'best'
    ),
    (
        # Search operator 2: Spiral Dynamics Optimization
        'spiral_dynamics_optimization',
        {
            'omega_max': 0.99,
            'alpha': 1.49618,
            'beta': 1.7
        },
        'best'
    ),
    (
        # Search operator 3: Particle Swarm Optimization
        'particle_swarm_optimization',
        {
            'w': 0.72984,
            'c1': 1.49618,
            'c2': 1.49618
        },
        'best'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=51)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Enhanced Hybrid Metaheuristic (EHM) combines three different operators to explore the search space more effectively.
# 'random_sample' provides a baseline exploration with a controlled population size and mutation rate.
# 'spiral_dynamics_optimization' simulates social behavior with adaptive dynamics, which helps in finding good solutions quickly.
# 'particle_swarm_optimization' ensures refined search around promising regions with updated parameters.
# By integrating these operators, the EHM aims to balance exploration and exploitation, leading to improved performance on the Rastrigin function.