# Name: Hybrid Metaheuristic for Optimization

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
            'scale': 0.3547010144703872,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0023626269199981485,
            'alpha': 0.09978515242026342,
            'beta': 1.8995874979846203,
            'dt': 1.6738940414467414
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand',
            'num_rands': 1,
            'factor': 0.5558309445970423
        },
        'probabilistic'
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
# The Hybrid Metaheuristic combines three different search operators: Random Search, Central Force Dynamic, and Differential Mutation. Each operator is configured with specific parameters to explore the solution space effectively. The Random Search uses a uniform distribution for exploration, while Central Force Dynamic simulates the movement of particles influenced by gravitational forces. Differential Mutation introduces diversity through random combinations of existing solutions. By combining these operators, the hybrid approach aims to balance exploration and exploitation, leading to more robust optimization results.

# The 'greedy' selector is used for Random Search to quickly find promising areas in the solution space. For Central Force Dynamic, the 'all' selector ensures that all agents consider the global best position to adjust their movement, promoting collective behavior. Differential Mutation utilizes a 'probabilistic' selector to adaptively choose the mutation strategy based on the population's characteristics.

# Running the metaheuristic for 30 iterations helps in assessing its performance and stability across different starting points, providing a more reliable estimate of the global optimum.