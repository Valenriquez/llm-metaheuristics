# Name: Hybrid Swarm-based Metaheuristic (HSMM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.benchmark_function(dimensions) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Particle Swarm Optimization
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial",
            'distribution': "uniform"
        },
        'probabilistic'
    ),
    (
        # Search operator 2: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': "uniform"
        },
        'greedy'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2 + (dimensions - 5) // 5) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# HSMM combines Particle Swarm Optimization (PSO) with Local Random Walk to explore the search space effectively. PSO is used to exploit the best-known regions of the solution space while L_RW helps in exploring new areas. The number of agents scales linearly with the dimension to ensure adequate exploration, especially for high-dimensional problems.