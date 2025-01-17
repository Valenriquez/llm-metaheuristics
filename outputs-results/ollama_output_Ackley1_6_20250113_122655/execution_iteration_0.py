# Name: Spiral Dynamic and Random Walk Hybrid

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
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
# This hybrid metaheuristic combines the Spiral Dynamic search operator with the Local Random Walk to explore the solution space more effectively. The Spiral Dynamic helps in moving towards regions of higher potential, while the Local Random Walk helps in fine-tuning the solutions by exploring the neighborhood. Together, they aim to find better solutions for the Ackley1 function in 6 dimensions.

# If an error occurs, address it as follows:
# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Ackley1_6_20250113_122655/execution_iteration_0.py", line 43, in <module>
#     print('x_best = {}, f_best = {}'.format(*met.get_solution()))
#                                              ^^^^^^^^^^^^^^^^^^
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 173, in get_solution
#     return self.historical['position'][-1], self.historical['fitness'][-1]
#            ~~~~~~~~~~~~~~~^^^^^^^^^^^^
# KeyError: 'position'
# .
# The error suggests that the historical data does not contain the 'position' key. Ensure that the metaheuristic is correctly logging and updating the historical data during each iteration.