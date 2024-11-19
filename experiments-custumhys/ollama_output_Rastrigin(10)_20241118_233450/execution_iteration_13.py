# Name: HybridMetaheuristicRastrigin
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 2.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 30.0,
            'sigma': 0.2
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines 'random_flight' and 'spiral_dynamic' operators. The 'random_flight' operator helps in exploring the search space
# effectively with a high scale factor to cover a wide area. The 'spiral_dynamic' operator then refines the search by utilizing the spiral path,
# which is known to converge towards the global optimum. Both operators use different selectors: 'greedy' for quick convergence and 'all'
# for exploring multiple paths simultaneously.
# The combination of these operators aims to achieve a balance between exploration and exploitation, making it suitable for complex optimization problems like Rastrigin's function.