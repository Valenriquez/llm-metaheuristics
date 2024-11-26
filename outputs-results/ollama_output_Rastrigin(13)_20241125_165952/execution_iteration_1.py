# Name: MetaHeuristicOptimized

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Search operator 1: Gravitational Search
# The gravitational search operator simulates the effect of gravity and attraction in finding solutions to optimization problems.
heur = [
    (
        'gravitational_search',
        {
            'gravity': 0.49186298165286546,
            'alpha': 0.010823605899875209
        },
        'metropolis'
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=500)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5)  # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the gravitational search operator with random sample. The gravitational search operator helps in exploring the solution space using the principles of gravity and attraction,
# while random sample is known for its ability to explore the entire search space without any bias or preference towards any particular solution. By combining these two operators, we aim to balance
# exploration and exploitation, leading to better convergence properties. The Metropolis selector ensures that even suboptimal solutions have a chance of being accepted,
# which aids in escaping local minima and finding the global minimum more efficiently.