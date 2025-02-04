# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.48879473273465485,
            'self_conf': 2.0264347321107756,
            'swarm_conf': 2.164754601859,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5908150666589067,
            'angle': 21.205110466277542,
            'sigma': 0.016006815320780035
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The HybridMetaheuristic combines the strengths of the swarm_dynamic and spiral_dynamic search operators. 
# Swarm_dynamic is effective in exploring large solution spaces, while spiral_dynamic helps in fine-tuning solutions.
# This hybrid approach aims to balance exploration and exploitation, potentially leading to better convergence rates and more robust solutions for the Rastrigin function.