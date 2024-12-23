# Name: Hybrid Swarm Optimization (HSO)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.20212091973573906,
            'self_conf': 1.0425870585646015,
            'swarm_conf': 0.8539383806591685,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
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
# HSO combines the strengths of swarm optimization and random sampling. The swarm_dynamic operator helps in exploring the solution space effectively, while the random_sample operator provides a way to escape local optima. This hybrid approach ensures a balance between exploration and exploitation, leading to better convergence and higher quality solutions.