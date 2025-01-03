# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.21525202339767957,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0032048307297649356,
            'alpha': 0.024756265515704968,
            'beta': 1.0411437146680746
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6252815272514172,
            'self_conf': 2.3482139274697342,
            'swarm_conf': 2.6158036185697546,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5004193422963226,
            'angle': 23.89533893071638,
            'sigma': 0.2535305839927631
        },
        'all'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# HybridMetaheuristic combines several different search operators to leverage their strengths. 
# The 'random_search' operator is used for exploration in the solution space.
# The 'central_force_dynamic' operator simulates a physical system with attractive forces, guiding solutions towards better areas.
# The 'swarm_dynamic' operator mimics the behavior of social animals, using collective intelligence to find optimal solutions.
# The 'spiral_dynamic' operator models the motion of spirals, which can help in efficiently exploring and converging on solutions.