# Name: BohachevskyMetaheuristic

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
            'scale': 0.04969096566537046,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.009979710504533262,
            'alpha': 0.09956868962981212,
            'beta': 1.539453683691987,
            'dt': 1.935104919105281
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
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
# This metaheuristic combines two search operators: 'random_search' and 'central_force_dynamic'. 
# The 'random_search' operator helps in exploring the solution space randomly, providing diverse initial solutions. 
# The 'central_force_dynamic' operator simulates the motion of particles under the influence of central forces, which can help in finding better solutions by guiding particles towards promising regions.
# The combination of these operators aims to leverage the exploration capabilities of random search with the exploitation capabilities of force-based dynamics, leading to an effective metaheuristic for solving optimization problems like Bohachevsky.