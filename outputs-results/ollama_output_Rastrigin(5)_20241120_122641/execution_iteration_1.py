# Name: Swarm Optimization Algorithm
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)
prob = fun.get_formatted_problem()

heur = [
    (
        'centralized_swarm_optimization',
        {
            'self_conf': 0.7349886025784094,
            'swarm_conf': 2.665057830738751,
            'version': 'inertial'
        },
        'roulette_wheel_selection'
    ),
    (
        'particle_swarm_optimization',
        {
            'radius': 0.5021392394715066,
            'angle': 0.20071936209445168,
            'sigma': 0.876130951284041
        },
        'elite_selection'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# This metaheuristic combines Centralized Swarm Optimization (CSO) and Particle Swarm Optimization (PSO).
# CSO uses a centralized approach with a modified version of the PSO algorithm. It incorporates parameters like self_conf, swarm_conf, 
# and version to ensure effective exploration and exploitation.
# The selector 'roulette_wheel_selection' is used for CSO as it ensures fair selection of agents for reproduction.
# For PSO, we use Particle Swarm Optimization (PSO) with parameters radius, angle, and sigma. This method efficiently handles high-dimensional problems.
# The selector 'elite_selection' prioritizes the fittest individuals for reproduction to maintain population diversity and convergence.
# The combination of both algorithms allows for a robust search strategy that balances exploration and exploitation effectively.