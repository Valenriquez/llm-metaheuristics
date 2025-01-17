# Name: Enhanced Adaptive Hybrid Metaheuristic (EAHM)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.5114368946114641,
            'self_conf': 2.8903584004228193,
            'swarm_conf': 1.6333491491525924,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.3237269837221421,
            'scale': 1.7386596656668696,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
# met.verbose = True # please comment this line
# met.run() # please comment this line

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
# The Enhanced Adaptive Hybrid Metaheuristic (EAHM) combines two effective metaheuristic operators: Swarm Dynamic and Local Random Walk. 
# The Swarm Dynamic operator is used to explore the solution space globally, while the Local Random Walk operator helps in fine-tuning the solutions locally.
# Both operators are driven by a probabilistic selector to ensure both exploration and exploitation phases of the search process.
# This hybrid approach aims to balance the strengths of both operators, leading to more robust convergence towards optimal solutions.