# Name: Hybrid Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.6414171360404639,
            'distribution': 'levy'
        },
        'metropolis'
    ),
    (
        # Search operator 2: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.08880282162543066,
            'alpha': 0.6176188117689989,
            'beta': 1.6314132682901883,
            'dt': 0.9074200839322611
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Differential Mutation
        'differential_mutation',
        {
            'expression': 'current-to-best',
            'num_rands': 1,
            'factor': 0.6352156315447706
        },
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
# The Hybrid Metaheuristic combines three different search operators: Random Search, Central Force Dynamic, and Differential Mutation. Each operator is configured with specific parameters to explore the solution space effectively. The use of 'metropolis' as a selector for Random Search helps in escaping local minima, while 'probabilistic' and 'greedy' selectors are used for Central Force Dynamic and Differential Mutation respectively to guide the search towards better solutions. This hybrid approach aims to leverage the strengths of each operator for comprehensive optimization of the problem.