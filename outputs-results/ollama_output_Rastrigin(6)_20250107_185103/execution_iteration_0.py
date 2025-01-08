# Name: Differential Evolution with Local Random Walk
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
        'differential_mutation',
        {
            'F': 0.5,
            'CR': 0.7
        },
        'best'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'random'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# Differential Evolution (DE) is a population-based optimization algorithm that uses mutation, crossover, and selection operations to iteratively improve a set of candidate solutions. It is particularly effective for global optimization problems.
# In this metaheuristic, we combine DE with Local Random Walk to enhance exploration and exploitation. The DE operator helps in escaping local minima by generating new candidates through mutation and crossover, while the Local Random Walk helps in fine-tuning the solution space locally. This hybrid approach aims to balance exploration and exploitation, leading to better convergence and more robust optimization results.
# The parameters F (mutation factor) and CR (crossover rate) for DE are set to 0.5 and 0.7 respectively, which have been found effective in many benchmark functions. The Local Random Walk operator uses a uniform distribution with a probability of 0.75 to ensure both exploration and exploitation balance.
# Running the metaheuristic multiple times (30 iterations) helps in assessing the stability and robustness of the solution. The final fitness array provides insights into the convergence behavior and performance of the algorithm across different runs.