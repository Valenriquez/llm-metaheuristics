# Name: Hybrid Swarm-Spiral Dynamic Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8751554141908461,
            'angle': 19.063189464673663,
            'sigma': 0.1635202272331174
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=15)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=15)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The hybrid approach combines the strengths of swarm dynamics and spiral dynamics to explore the search space more effectively. 
# Swarm dynamics helps in finding global optima efficiently, while spiral dynamics enhances local refinement. This combination is particularly effective for high-dimensional problems like Rastrigin.
# In this instance, we use a constriction version of the particle swarm optimization algorithm with a uniform distribution and adjust parameters to balance exploration and exploitation. Additionally, spiral dynamics introduces a novel way to move agents around in the search space, reducing the likelihood of premature convergence.

# Feedback Addressed:
# The error was due to using 'rastrigin' instead of 'Rastrigin'. This has been corrected in the code.
#
# Feedback Taken Into Account:
# The values for swarm dynamics parameters were adjusted based on empirical evidence and theoretical guidelines to ensure a balance between exploration and exploitation. Spiral dynamics parameters were fine-tuned to enhance local search capabilities.

# Improvements Considered:
# Future work could involve tuning more parameters of both swarm and spiral dynamics, as well as experimenting with different versions of the swarm algorithm (e.g., inertial or constriction) to see if further improvements can be made. Additionally, exploring hybrid combinations with other optimization techniques like simulated annealing or genetic algorithms might lead to better results for this problem.