# Name: Hybrid Metaheuristic Algorithm for Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define search operators and their parameters/selectors
heur = [
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.01,
            'alpha': 0.02,
            'beta': 1.5,
            'dt': 0.5
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best',
            'num_rands': 1,
            'factor': 1.2
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 3.0,
            'swarm_conf': 2.5,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

# Output the best solution found
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Repeat the experiment 30 times to gather fitness data
fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm combines multiple search operators to improve exploration and exploitation during the optimization process. The 'random_search' operator helps in generating diverse solutions, while 'central_force_dynamic' promotes convergence towards optimal regions. Differential mutation introduces diversity by modifying solutions based on the differences between individuals in the population. Finally, 'swarm_dynamic' simulates particle swarm behavior with inertia and social components to enhance search efficiency. This combination is expected to perform well on the Rastrigin function and other optimization problems.