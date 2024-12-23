# Name: Hybrid Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.3693425972851212,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.007509222662017407,
            'alpha': 0.002215583983671063,
            'beta': 1.0172747684482208,
            'dt': 0.2515441635118518
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 1.3242436287544117
        },
        'probabilistic'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 0.002215583983671063,
            'beta': 1.0172747684482208,
            'gamma': 152.34635863484766
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

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
# The Hybrid Metaheuristic combines four different search operators: random search, central force dynamic, differential mutation, and firefly dynamic. Each operator is designed to bring unique dynamics to the search process, enhancing exploration and exploitation. The selection of operators ensures a balanced approach, starting with random search for broad exploration and gradually moving towards more sophisticated dynamic models. This design is grounded in the provided data, using parameters that are directly applicable and meaningful to the search space. By running the metaheuristic 30 times, we can observe the stability and effectiveness of the hybrid approach across different iterations.