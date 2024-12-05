# Name: Random Search with Spiral Dynamic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'random_sample'
    ),
    (
        'random_sample',
        {},
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The metaheuristic is based on a combination of Spiral Dynamic search operator to guide the agents in a spiral motion around the optimal solution, and Random Sample to ensure exploration. This hybrid approach aims to balance exploitation and exploration effectively.

# The use of the Spiral Dynamic operator helps the agents converge towards local optima by adjusting their position based on the radius, angle, and sigma parameters. The Random Sample operator adds diversity to the search process, preventing premature convergence and allowing the algorithm to escape local minima more often.

# The choice of selectors such as 'random_sample' and 'probabilistic' ensures that each operator is used appropriately throughout the execution of the metaheuristic. The random_sample selector helps in generating a diverse set of solutions, while the probabilistic selector controls the probability distribution from which new solutions are sampled.

# By combining these operators and selectors, the resulting metaheuristic can effectively navigate complex optimization landscapes, leading to better performance on benchmark functions like Rastrigin.