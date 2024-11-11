# Name: Ackley_Combinator
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 0.7,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'uniform',
            'beta': 2.5
        },
        'all'
    ),
    (
        'genetic_mutation',
        {
            'scale': 0.8,
            'elite_rate': 0.15,
            'mutation_rate': 0.3,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different operators: gravitational search, random flight, and genetic mutation. 
# The gravitational search operator is used to guide the search towards better solutions. 
# The random flight operator is used to explore new regions of the search space. 
# The genetic mutation operator is used to introduce diversity into the population.
# By combining these three operators, we can create a more robust and effective metaheuristic algorithm.

# Name: Ackley_Swarm
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.05
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 1.2,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.05,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different operators: spiral dynamic, local random walk, and genetic mutation. 
# The spiral dynamic operator is used to guide the search towards better solutions. 
# The local random walk operator is used to explore new regions of the search space. 
# The genetic mutation operator is used to introduce diversity into the population.
# By combining these three operators, we can create a more robust and effective metaheuristic algorithm.

# Name: Ackley_Combinator_2
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    (
        'random_flight',
        {
            'scale': 1.5,
            'distribution': 'gaussian',
            'beta': 2.8
        },
        'metropolis'
    ),
    (
        'genetic_mutation',
        {
            'scale': 0.9,
            'elite_rate': 0.2,
            'mutation_rate': 0.35,
            'distribution': 'levy'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different operators: gravitational search, random flight, and genetic mutation. 
# The gravitational search operator is used to guide the search towards better solutions. 
# The random flight operator is used to explore new regions of the search space. 
# The genetic mutation operator is used to introduce diversity into the population.
# By combining these three operators, we can create a more robust and effective metaheuristic algorithm.