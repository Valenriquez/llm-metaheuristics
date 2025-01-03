# Name: Hybrid Metaheuristic Algorithm
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.8,
            'dt': 0.5
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 1,
            'factor': 0.9
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=500)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1500, num_agents=12)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines three different search operators: Random Search, Central Force Dynamic, and Differential Mutation. Each operator is given a set of parameters tailored for the Rastrigin function. The Random Search operator uses a small scale and uniform distribution to explore the solution space randomly but more thoroughly. The Central Force Dynamic operator is designed to move agents towards better solutions, using a gravity-like force with adaptive alpha and beta values. The Differential Mutation operator introduces diversity by mutating individuals based on differences between population members, enhancing exploration and exploitation capabilities. The Metropolis selector is used for Random Search, allowing it to escape local minima more effectively. The 'all' selector for Central Force Dynamic ensures that all agents are influenced simultaneously, promoting global convergence. The probabilistic selector for Differential Mutation increases the likelihood of generating new and potentially better solutions.
# Running the metaheuristic 30 times helps in assessing its robustness and providing a better understanding of its performance across different starting points and conditions.