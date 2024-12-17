# Name: Advanced Metaheuristic for Multi-dimensional Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Determine the number of agents based on the dimension size
num_agents = max(2, prob['dimension'] * 5)

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'probabilistic'
    )
]

# For dimensions greater than or equal to 3, add more operators and selectors
if prob['dimension'] >= 3:
    heur.extend([
        (
            'differential_mutation',
            {
                'expression': 'rand',
                'num_rands': 1,
                'factor': 1.0
            },
            'metropolis'
        ),
        (
            'firefly_dynamic',
            {
                'distribution': 'gaussian',
                'alpha': 1.0,
                'beta': 1.0,
                'gamma': 100.0
            },
            'greedy'
        ),
        (
            'gravitational_search',
            {
                'gravity': 1.0,
                'alpha': 0.02
            },
            'all'
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
            'probabilistic'
        )
    ])

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines several search operators including random search, central force dynamic, differential mutation, firefly dynamics, gravitational search, and swarm dynamics. Each operator is paired with a suitable selector to ensure diversity in the population and enhance exploration and exploitation of the solution space.
# The number of agents is dynamically adjusted based on the dimension size to optimize performance. This approach ensures that smaller problems are solved efficiently while larger problems receive sufficient resources for better convergence.