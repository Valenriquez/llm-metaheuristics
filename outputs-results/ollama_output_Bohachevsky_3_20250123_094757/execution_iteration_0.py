# Name: Hybrid Metaheuristic for Bohachevsky Function with Modified Parameters

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.011706613404095012,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.009968364918734237,
            'alpha': 0.015580551393589062,
            'beta': 1.5380136991963815,
            'dt': 1.7901299585244685
        },
        'all'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines two different search operators: Random Search and Central Force Dynamics.
# The Random Search operator helps in exploring the solution space randomly with a Levy distribution, which is beneficial for escaping local minima.
# The Central Force Dynamics operator mimics the behavior of particles moving under central forces with specific parameters that promote convergence towards the global optimum.
# The combination of these operators allows the metaheuristic to balance exploration and exploitation effectively, enhancing its performance on the Bohachevsky function.