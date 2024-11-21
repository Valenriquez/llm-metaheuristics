# Name: Hybrid Metaheuristic with Spiral and Swarm Dynamics
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The metaheuristic named "Hybrid Metaheuristic with Spiral and Swarm Dynamics" combines two search operators: 
# 'spiral_dynamic' and 'swarm_dynamic'. These operators were chosen based on their effectiveness in navigating the search space.
# The 'spiral_dynamic' operator guides the search along a spiral trajectory, which can be particularly useful for fine-tuning solutions near optima. 
# The 'swarm_dynamic' operator mimics the behavior of social animals, such as birds and fish, to explore large solution spaces and find good solutions quickly.
# Both operators use greedy selectors to ensure that the best individuals are retained for the next iteration, enhancing convergence.
# The metaheuristic is designed to be robust and efficient for solving optimization problems, especially those with complex landscapes like Rastrigin's function.