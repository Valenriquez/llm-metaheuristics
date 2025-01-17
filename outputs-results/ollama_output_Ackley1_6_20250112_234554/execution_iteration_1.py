# Name: Ackley1_6_Metaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 25,
            'sigma': 0.05
        },
        'metropolis'
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
# The Ackley1 function is a well-known benchmark function used to evaluate the performance of optimization algorithms. 
# In this metaheuristic, we have combined two search operators: 'random_sample' and 'spiral_dynamic'.
# The 'random_sample' operator randomly samples solutions from the solution space, providing diversity.
# The 'spiral_dynamic' operator uses a spiral-based movement strategy to explore the solution space more efficiently.
# We also use the 'metropolis' selector for the 'spiral_dynamic' operator, allowing it to escape local minima and continue exploring.
# By running the metaheuristic 30 times with different initializations, we can gather statistical data on the algorithm's performance.