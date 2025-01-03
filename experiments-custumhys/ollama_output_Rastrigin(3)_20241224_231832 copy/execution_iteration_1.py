# Name: Hybrid Metaheuristic with Diverse Operators

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {},
        'greedy'
    ),
    (
        # Search operator 2: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.6983976096916454,
            'scale': 0.5331512334242134,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.5037265752814947,
            'angle': 14.398662107665004,
            'sigma': 0.15416507045880048
        },
        'greedy'
    ),
    (
        # Search operator 4: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.7935500696509203,
            'self_conf': 1.3732995116039555,
            'swarm_conf': 2.61154188020552,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines four different search operators: Random Sample, Local Random Walk, Spiral Dynamic, and Swarm Dynamic. The operators are selected based on their ability to explore the solution space effectively in different ways.
# - **Random Sample**: Provides a random exploration of the solution space.
# - **Local Random Walk**: Explores the neighborhood of the current solution to find improvements.
# - **Spiral Dynamic**: Explores in a spiral manner, which can help in covering areas that are not immediately accessible through simple linear movements.
# - **Swarm Dynamic**: Mimics the behavior of a swarm, allowing multiple agents to work together and share information.

# The operators are selected with different selection strategies (greedy, metropolis, probabilistic) to balance exploration and exploitation. This approach aims to leverage the strengths of each operator and improve the overall performance of the metaheuristic.