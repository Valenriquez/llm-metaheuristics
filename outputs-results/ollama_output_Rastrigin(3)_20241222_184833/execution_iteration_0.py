# Name: Hybrid Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the search operators and their parameters
heur = [
    (
        'random_search',
        {
            'scale': 0.39464066818068844,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.1465837371395332,
            'alpha': 0.5001000792133685,
            'beta': 2.494907619499824,
            'dt': 0.843324523849551
        },
        'greedy'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 2,
            'factor': 2.6522523571867405
        },
        'probabilistic'
    )
]

# Initialize the metaheuristic with the problem and operators
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic multiple times to get an average result
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# This hybrid metaheuristic combines three different search operators: random search, central force dynamic, and differential mutation. The random search operator helps in exploring the solution space while avoiding local optima. The central force dynamic operator simulates physical forces between agents to guide them towards better solutions. Differential mutation is used to explore the neighborhood of current solutions by using the difference between two randomly chosen solutions. The combination of these operators ensures a balance between exploration and exploitation, making it suitable for optimizing complex problems like Rastrigin's function. Running the metaheuristic multiple times allows us to get an average result, which provides a more robust estimation of the optimal solution.