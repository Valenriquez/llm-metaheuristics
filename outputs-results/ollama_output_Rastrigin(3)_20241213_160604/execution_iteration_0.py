# Name: Adaptive Spiral Dynamic Optimization Metaheuristic (ASDOM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Determine number of agents based on dimension size
num_agents = self.dimensions + 2 if self.dimensions >= 5 else 2

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7 if self.dimensions < 5 else 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial' if self.dimensions < 5 else 'constriction',
            'distribution': 'uniform'
        },
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# ASDOM combines Spiral Dynamic Optimization (Spiral-DO) with Particle Swarm Optimization (PSO). The Spiral-DO operator is effective for fine-grained search in high dimensions, while PSO efficiently explores the solution space. Together, they offer a powerful approach to solving complex benchmark functions.
    
# Additionally, the number of agents is dynamically adjusted based on the dimensionality to ensure sufficient exploration and exploitation capabilities across different problem landscapes.