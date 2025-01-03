# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.9480267766386352,
            'self_conf': 2.532928222685076,
            'swarm_conf': 2.6085275807614825,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6075879498660249,
            'angle': 18.336801276968053,
            'sigma': 0.06082869697490708
        },
        'metropolis'
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
# This hybrid metaheuristic combines three different operators: random_sample, swarm_dynamic, and spiral_dynamic. The 'random_sample' operator is used to explore the solution space randomly, which helps in finding new promising regions. The 'swarm_dynamic' operator simulates the behavior of a group of agents (particles) moving towards the best solutions found so far, which can efficiently converge to optimal solutions. The 'spiral_dynamic' operator introduces a spiral-like movement, which can help escape local optima and explore different parts of the solution space. The combination of these operators allows for a more comprehensive search, leading to better results on complex benchmark functions like Rastrigin.

# Addressing the error:
# The error occurred because the 'differential_mutation' operator received an unexpected keyword argument 'cr'. This suggests that there might be a mismatch between the expected parameters and the actual arguments being passed. To resolve this issue, ensure that the parameters for each operator are correctly specified according to their definitions in the `metaheuristic.py` file. Additionally, double-check the version of the metaheuristic library being used, as different versions may have changes or updates in function signatures.