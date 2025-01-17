# Name: Particle Swarm Optimization with Genetic Crossover (PSO-GC)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.3965650716732098,
            'self_conf': 2.6566742927951674,
            'swarm_conf': 2.564003019391465,
            'version': "constriction",
            'distribution': "gaussian"
        },
        'probabilistic'
    ),
    (
        'genetic_crossover',
        {
            'pairing': "tournament_2_100",
            'crossover': "blend",
            'mating_pool_factor': 0.4800131870285026
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# PSO-GC is a hybrid approach combining the strengths of Particle Swarm Optimization (PSO) and Genetic Crossover. The swarm_dynamic operator helps particles explore the search space efficiently, while genetic_crossover ensures diversity within the population, enhancing the exploration capability. This combination aims to balance exploitation and exploration, leading to better convergence towards the optimal solution with improved robustness and effectiveness in handling global optimization problems like the Rastrigin function.