# Name: Hybrid Metaheuristic for Optimization Problems
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines gravitational search and random sampling. 
# The gravitational search operator is used to attract solutions towards the best known positions, 
# while random sampling helps in exploring new regions of the solution space. 
# The 'greedy' selector ensures that the most promising solutions are selected for further exploration. 
# This combination aims to balance exploitation (using gravitational search) and exploration (using random sampling), enhancing the overall performance of the optimization process.