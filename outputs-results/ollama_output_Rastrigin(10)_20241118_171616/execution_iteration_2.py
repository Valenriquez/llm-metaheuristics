# Name: Hybrid Metaheuristic with Spiral Dynamic and Local Random Walk
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
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
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
# This metaheuristic combines the Spiral Dynamic (spiral_dynamic) operator with the Local Random Walk (local_random_walk) operator.
# The Spiral Dynamic operator is used to explore the solution space in a spiral manner, which is effective for multimodal optimization problems.
# The Local Random Walk operator is used to refine and exploit promising regions of the solution space, helping to converge faster towards the global optimum.
# The combination of these operators allows for a balance between exploration and exploitation, making the metaheuristic well-suited for the Rastrigin function, which has many local minima.