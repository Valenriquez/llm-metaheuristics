# Name: Random Walk and Local Search Hybrid Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(5)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',  # Search operator 2: Random Sample
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=500)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5)  # Adjusting number of agents for a 5-dimensional problem
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines the Local Random Walk and Random Sample operators. The Local Random Walk operator is used to explore the neighborhood of current solutions, while the Random Sample operator helps in diversifying the search space. This combination allows for a balance between exploration and exploitation, which is crucial in optimizing complex problems like Ackley1 with multiple optima.