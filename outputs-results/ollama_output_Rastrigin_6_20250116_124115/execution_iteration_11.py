# Name: Hybrid Evolutionary Search with Dynamic Operators

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 1.1734950823239607,
            'distribution': 'gaussian',
            'beta': 3.8475290098274657
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5059279386406621,
            'self_conf': 2.3785144368154962,
            'swarm_conf': 2.9714595681872025,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8223601782778941,
            'angle': 27.35122483118511,
            'sigma': 0.2392965015541771
        },
        'all'
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
# Hybrid Evolutionary Search with Dynamic Operators combines the strengths of three different metaheuristic operators: random flight, swarm dynamic, and spiral dynamic. Each operator is configured with parameters that are tuned to work well for the Rastrigin function in a 6-dimensional space.
#
# The random flight operator helps escape local minima by making large, random moves.
#
# The swarm dynamic operator simulates the behavior of social insects, encouraging agents to explore and exploit solutions collaboratively.
#
# The spiral dynamic operator uses a mathematical model to guide agents towards promising regions of the search space in a more controlled manner.
#
# By integrating these operators, the hybrid approach aims to balance exploration and exploitation, potentially leading to better convergence and higher quality solutions for the Rastrigin function.