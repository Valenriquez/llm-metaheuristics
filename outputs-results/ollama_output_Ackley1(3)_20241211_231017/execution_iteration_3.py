# Name: Enhanced Ackley Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'hill_climb',
        {
            'step_size': 0.1,
            'max_iterations': 10
        },
        'probabilistic'
    ),
    (
        'annealing',
        {
            'initial_temperature': 1.0,
            'cooling_rate': 0.95,
            'min_temperature': 0.01
        },
        'simulated_annealing'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)  # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Enhanced Ackley Metaheuristic combines the `random_sample` operator with a probabilistic selection strategy from the `hill_climb` operator. Additionally, it incorporates an `annealing` component with a simulated annealing selector. This hybrid approach aims to balance exploration and exploitation while enhancing the search space coverage.
#
# In each iteration, 30 runs of the metaheuristic are performed, adjusting the number of agents based on the dimensionality (2 agents for a 3-dimensional problem). The fitness data is collected over these iterations to evaluate the effectiveness of the enhanced Ackley metaheuristic.