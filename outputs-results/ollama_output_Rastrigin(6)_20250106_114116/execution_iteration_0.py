# Name: Hybrid Metaheuristic with Random Search and Differential Mutation

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.8549764808764913,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best',
            'num_rands': 2,
            'factor': 0.11613523558771505
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines a local search operator (Random Search) with a global search operator (Differential Mutation). The Random Search operator helps to explore the solution space more thoroughly, while the Differential Mutation operator promotes exploration by using the difference between two or more solution vectors. The combination of these operators allows for a balance between exploitation and exploration, potentially leading to better solutions and faster convergence.