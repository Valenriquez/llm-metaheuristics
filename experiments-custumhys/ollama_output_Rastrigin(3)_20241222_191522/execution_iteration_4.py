# Name: Hybrid Metaheuristic for Rastrigin Function Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
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
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

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
# This hybrid metaheuristic combines three different search operators to optimize the Rastrigin function. The first and third operators are `random_sample`, which is a simple random sampling operator that helps in exploring the solution space initially. The second operator is `swarm_dynamic`, which is an implementation of particle swarm optimization (PSO). PSO is a population-based stochastic optimization algorithm inspired by social behavior patterns, such as bird flocking or fish schooling. By combining these operators, we aim to leverage both exploration and exploitation capabilities, leading to more efficient convergence towards the optimal solution.

# The selection of `greedy` selector for each operator indicates that the next step in the search is chosen based on the best immediate improvement available, which helps in quickly finding promising regions of the solution space. The number of iterations and agents are set to 100 and 10 respectively, which can be adjusted based on specific problem requirements and computational resources.

# Addressing potential errors:
# If an error occurs due to incorrect operator parameters or unsupported operators, it is important to verify that all parameters match the expected input for each operator. Additionally, ensuring that the `selector` chosen matches the operator's expectations (e.g., 'greedy' for deterministic selection) can help resolve issues.