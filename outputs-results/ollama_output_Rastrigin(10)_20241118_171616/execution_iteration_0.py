# Name: Hybrid Metaheuristic with Random Sample and Local Random Walk

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
    (  # Search operator 1
        'random_sample',
        {},
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines two operators: 'random_sample' and 'local_random_walk'. 
# 'Random_sample' provides a broad exploration of the search space by selecting random solutions, ensuring global diversity.
# 'Local_random_walk' refines the solution through probabilistic local modifications, making fine adjustments to improve convergence. 
# This combination allows for both extensive exploration and intensive exploitation, enhancing the overall performance on complex optimization problems like Rastrigin's function.