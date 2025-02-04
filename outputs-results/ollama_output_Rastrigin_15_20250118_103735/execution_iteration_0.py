# Name: HarmonySearchWithDifferentialMutation

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.252152768753621,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'factor': 0.5875438876347718
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines the random search operator with a differential mutation operator. The random search helps to explore the solution space broadly using Gaussian distribution, while the differential mutation is used to exploit promising regions by making incremental changes based on the best solutions. The combination of these two operators allows for a balance between exploration and exploitation, which can be beneficial for complex optimization problems like Rastrigin's function. The metropolis selector is used for the random search operator, allowing it to escape local minima, while the probabilistic selector is used for differential mutation, encouraging more diverse search by accepting good solutions with some probability. Running the metaheuristic multiple times helps in assessing the robustness and performance of the solution across different initializations.