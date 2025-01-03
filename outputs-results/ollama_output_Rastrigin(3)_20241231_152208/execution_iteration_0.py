# Name: Hybrid Metaheuristic Algorithm for Rastrigin Function Optimization

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
    (  # Search operator 1: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'greedy'
    ),
    (
        # Search operator 2: Differential Mutation
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 1,
            'factor': 1.0
        },
        'all'
    ),
    (
        # Search operator 3: Random Sample
        'random_sample',
        {},
        'probabilistic'
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
# The Hybrid Metaheuristic Algorithm combines three different search operators to optimize the Rastrigin function. 
# 1. Central Force Dynamic is used to simulate the influence of gravitational forces on particles in a system, guiding them towards lower energy states.
# 2. Differential Mutation introduces diversity into the population by mutating individuals based on differences between other individuals, helping to escape local optima.
# 3. Random Sample ensures that new solutions are generated randomly, maintaining exploration and preventing premature convergence.
# The use of different selectors with varying levels of randomness (greedy, all, probabilistic) helps balance exploitation and exploration throughout the optimization process.