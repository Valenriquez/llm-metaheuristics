# Name: HybridMetaheuristic

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
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.01,
            'alpha': 0.05,
            'beta': 2.0,
            'dt': 1.0
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best',
            'num_rands': 2,
            'factor': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

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
# The HybridMetaheuristic combines three different search operators: Random Search, Central Force Dynamics, and Differential Mutation. Each operator is configured with specific parameters to enhance its performance on the Rastrigin function.
#
# 1. **Random Search**: This operator helps in exploring the solution space by randomly selecting points. It uses a small scale and uniform distribution to ensure thorough exploration.
# 2. **Central Force Dynamics**: This operator mimics the behavior of particles moving under the influence of central forces. By adjusting gravity, alpha, beta, and time step, it efficiently converges towards optimal solutions.
# 3. **Differential Mutation**: This operator uses a mutation strategy based on differences between randomly chosen individuals in the population. The use of 'rand-to-best' ensures that it focuses on improving upon the best individual found so far.
#
# Together, these operators provide a robust search process that balances exploration and exploitation, making the HybridMetaheuristic well-suited for solving complex optimization problems like the Rastrigin function.