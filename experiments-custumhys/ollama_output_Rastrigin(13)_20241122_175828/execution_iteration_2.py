# Name: Improved Swarm Dynamic Metaheuristic
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
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'gaussian'
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
# The Improved Swarm Dynamic Metaheuristic combines the Swarm Dynamic operator with a random sampling operator to enhance exploration and exploitation. The Swarm Dynamic operator is configured with specific parameters such as 'factor', 'self_conf', 'swarm_conf', 'version', and 'distribution' to adaptively adjust the movement of agents in the search space. The random sampling operator helps in diversifying the search by randomly selecting solutions, which can be crucial for avoiding local optima. This combination is expected to improve the overall performance and robustness of the metaheuristic on the Rastrigin function with 13 dimensions.
#