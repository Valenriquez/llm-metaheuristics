# Name: Enhanced Metaheuristic with Adaptive Search Operators and Selectors
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  
        'local_random_walk',  # Operator 1
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'  # Using 'all' selector for higher dimensionality
    ),
    (
        'random_sample',  # Operator 2
        {},
        'all'  # Using 'all' selector for higher dimensionality
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines two operators: 'local_random_walk' and 'random_sample'. 
# The 'local_random_walk' operator helps the algorithm explore the neighborhood of the current solution, ensuring that it gets stuck in local minima.
# The 'random_sample' operator adds randomness to the search process, helping to escape from local optima and explore other parts of the solution space.
# Both operators are paired with an 'all' selector, which allows them to adaptively choose the best operator for each iteration based on the problem's characteristics.