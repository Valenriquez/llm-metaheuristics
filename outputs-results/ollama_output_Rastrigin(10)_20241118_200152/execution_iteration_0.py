# Name: Hybrid Metaheuristic with Random Sample and Local Random Walk
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {
            'parameter1': None,
            'parameter2': None,
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian',
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Hybrid Metaheuristic with Random Sample and Local Random Walk combines two different search operators: 
# 'random_sample' and 'local_random_walk'. The random sample operator helps in exploring the solution space by randomly selecting points, 
# which can help in finding good initial solutions. The local random walk operator is used to explore the neighborhood of a given solution, 
# allowing for fine-tuning of the solution. This hybrid approach aims to balance exploration and exploitation, potentially leading to better convergence.