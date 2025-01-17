# Name: Hybrid Metaheuristic for Rastrigin Function

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
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.9982504354703299,
            'self_conf': 2.555643761814626,
            'swarm_conf': 2.590324847994862,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5698630164263059,
            'angle': 20.93164648373638,
            'sigma': 0.13862896565816035
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=20)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines the strengths of three different search operators: random sampling, swarm dynamics, and spiral dynamics. This combination aims to explore the solution space more effectively while maintaining a balance between exploration and exploitation. Random sampling helps in diversifying the search process, swarm dynamics ensures that promising regions are explored thoroughly with the specified parameters for constriction and uniform distribution, and spiral dynamics adds an element of novelty by moving agents in a spiral pattern around the best solutions found so far with specific radius, angle, and sigma values. The use of different selectors (greedy, metropolis, probabilistic) further enhances the robustness of the metaheuristic by adapting to the characteristics of the problem.