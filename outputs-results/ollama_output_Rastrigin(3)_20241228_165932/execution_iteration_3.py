# Name: Hybrid Metaheuristic with Diverse Strategies

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.6817263708201861,
            'scale': 1.344935268219217,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5032115631742173,
            'angle': 23.893369547430297,
            'sigma': 0.017188767511193553
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines different search operators to explore the solution space effectively. The random_sample operator provides a broad initial exploration, local_random_walk focuses on local improvements with a certain probability, and spiral_dynamic helps in escaping local optima by following a spiral path. The use of the metropolis selector ensures that the solutions are accepted based on their quality relative to the current best solution, promoting diversification and intensification.

# Note:
# If you encounter an error similar to the one mentioned, ensure that all operators used are correctly implemented in the `metaheuristic.py` module and that they accept the parameters specified in the heuristic list. Adjust the parameters as needed based on the specific requirements of each operator.