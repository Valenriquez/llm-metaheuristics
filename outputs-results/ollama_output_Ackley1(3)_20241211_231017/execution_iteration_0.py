# Name: Ackley Hybrid Metaheuristic
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
        'swarm_dynamic',
        {
            'factor': 0.26539548694017,
            'self_conf': 2.0056335037496824,
            'swarm_conf': 0.9035840517873581,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 23.84400382909626,
            'sigma': 0.06481045949545582
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5)  # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Ackley Hybrid Metaheuristic combines the `random_sample` operator with a probabilistic selection strategy from the `swarm_dynamic` operator. Additionally, it incorporates a `spiral_dynamic` component with a metropolis selector. This hybrid approach aims to balance exploration and exploitation while enhancing the search space coverage.
    
# In each iteration, 30 runs of the metaheuristic are performed, adjusting the number of agents based on the dimensionality (5 agents for a 3-dimensional problem). The fitness data is collected over these iterations to evaluate the effectiveness of the hybrid metaheuristic.