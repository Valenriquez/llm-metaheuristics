# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.07565669449109641,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7902180636478461,
            'self_conf': 2.9999585549717787,
            'swarm_conf': 2.9451976105725395,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.787647387858027,
            'angle': 23.15262347051552,
            'sigma': 0.12455744889404319
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The HybridMetaheuristic combines three different search operators: Random Search, Swarm Dynamics, and Spiral Dynamics. Each operator has been configured with specific parameters to improve their performance on the Rastrigin function.
# The 'random_search' operator uses a small scale and uniform distribution for exploring the solution space randomly. This helps in escaping local optima.
# The 'swarm_dynamic' operator is designed to mimic the behavior of particle swarms, using inertia, self-confidence, and swarm confidence parameters. It operates with a uniform distribution for better convergence.
# The 'spiral_dynamic' operator uses a spiral pattern to explore the solution space efficiently. It has a moderate radius, angle, and sigma value.
# The combination of these operators allows for a diverse search strategy that balances exploration and exploitation. The use of metropolis and probabilistic selectors helps in accepting new solutions based on their fitness values, enhancing the overall performance of the metaheuristic.
# The algorithm runs for 1000 iterations with 30 agents for each run to ensure robustness and gather enough data for analysis.