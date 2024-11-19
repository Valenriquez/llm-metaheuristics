# Name: Improved Gravitational Search Algorithm with Random Flight and Local Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The proposed metaheuristic algorithm combines three different search operators: Gravitational Search Algorithm (GSA), Random Flight (RF), and Local Random Walk (LRW). 
# GSA is effective for finding global optima, while RF helps in exploring new regions of the solution space. LRW ensures fine-tuning around the current best solutions.
# The use of 'all' selector allows each operator to adapt to different stages of the optimization process, ensuring a balanced exploration and exploitation strategy.