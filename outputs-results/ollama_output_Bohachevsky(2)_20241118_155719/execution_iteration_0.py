# Name: Random Search with Spiral Dynamic and Local Random Walk
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

heur = [
    ('swarm_dynamic',
        {
            'factor': 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial",
            'distribution': "uniform"
        },
        'probabilistic'
    ),
    ('spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    ('local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': "uniform"
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the strengths of three different search operators to explore the solution space effectively. 
# The 'swarm_dynamic' operator is used initially to provide a broad exploration, utilizing particle swarm optimization principles.
# The 'spiral_dynamic' operator is then introduced to refine the search by leveraging spiral dynamics for a more focused exploration.
# Finally, the 'local_random_walk' operator helps in escaping local optima by performing random walks within the solution space.
# The combination of these operators ensures that the algorithm can efficiently navigate through complex landscapes and find near-optimal solutions.