# Name: Hybrid Swarm-Spiral Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.4559119122576877,
            'self_conf': 2.885114996229021,
            'swarm_conf': 0.6866532547798144,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7997321761150928,
            'angle': 1.0102645723350037,
            'sigma': 0.8531097767389534
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The hybrid Swarm-Spiral Metaheuristic combines the strengths of both swarm intelligence and spiral dynamics to explore the search space more effectively. The swarm_dynamic operator uses an inertial-based version with parameters tuned for global exploration, while the spiral_dynamic operator helps in fine-tuning the solution around the best found area.
# This combination aims to balance between exploration and exploitation, leading to potentially better convergence rates and solutions for complex optimization problems like Rastrigin(13).