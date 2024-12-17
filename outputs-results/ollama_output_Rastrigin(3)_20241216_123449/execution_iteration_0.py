# Name: Hybrid Multi-Operator Adaptive Search (HMOAS)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Flight Operator
        'RandomFlight',
        {
            'scale': 1.5,
            'distribution': 'levy',
            'beta': 1.3
        },
        'Probabilistic'
    ),
    (
        # Search operator 2: Local Random Walk Operator
        'LocalRandomWalk',
        {
            'probability': 0.6,
            'scale': 1.2,
            'distribution': 'gaussian'
        },
        'Metropolis'
    ),
    (
        # Search operator 3: Spiral Dynamic Operator
        'SpiralDynamic',
        {
            'radius': 0.85,
            'angle': 25.0,
            'sigma': 0.12
        },
        'Greedy'
    ),
    (
        # Search operator 4: Swarm Dynamic Operator
        'SwarmDynamic',
        {
            'factor': 0.8,
            'self_conf': 2.6,
            'swarm_conf': 2.7,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'Probabilistic'
    ),
    (
        # Search operator 5: Spiral Dynamic Operator
        'SpiralDynamic',
        {
            'radius': 0.85,
            'angle': 25.0,
            'sigma': 0.12
        },
        'Greedy'
    )
]

# Determine the number of agents based on dimension size
num_agents = {self.dimensions} if {self.dimensions} <= 3 else {self.dimensions} * {self.dimensions}

met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=num_agents)
met.verbose = True

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
# The Hybrid Multi-Operator Adaptive Search (HMOAS) metaheuristic combines multiple optimization strategies including Random Flight, Local Random Walk, Spiral Dynamic, and Swarm Dynamic operators. Each operator is assigned a different selector to control its application: Probabilistic for exploration, Metropolis for probabilistic acceptance of worse solutions, and Greedy for exploitation.
# The number of agents is dynamically determined based on the problem's dimension size to ensure adequate exploration capabilities.
# The heuristic is run multiple times to evaluate its effectiveness in finding better global solutions compared to using a single operator or selector.