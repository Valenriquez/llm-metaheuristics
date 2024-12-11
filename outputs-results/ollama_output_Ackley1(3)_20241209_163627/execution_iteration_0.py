# Name: Particle Swarm Optimization with Constriction Version and Uniform Distribution
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(5)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Increase the number of agents based on dimension
num_agents = 5  # For a 5-dimensional problem

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7499114446143524,
            'self_conf': 2.1300661073310345,
            'swarm_conf': 2.4414287004906425,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'gaussian_mutation',
        {
            'sigma': 0.1
        },
        'acceptance_based'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines swarm dynamics with a constriction version for better convergence.
# The Gaussian mutation operator helps in exploring the search space effectively.
# The number of agents is adjusted based on the problem dimension to ensure thorough exploration.