# Name: Multi-Objective Particle Swarm Optimization with Adaptive Learning Rate
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rosenbrock(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

num_agents = 7  # Number of agents depends on the dimension (2 + (dimension - 2) // 4)
heur = [
    (
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
        'random_flight',
        {
            'scale': 1.0,
            'distribution': "uniform",
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the Swarm Dynamic operator with the Random Flight operator to explore the solution space effectively. The swarm dynamic operator helps in converging towards a solution while the random flight operator aids in escaping local minima. The probabilistic selector ensures that both operators are considered randomly at each iteration, enhancing exploration and exploitation capabilities.