# Name: Constriction Swarm Metaheuristic (CSM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley15()  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.5676482212040213,
            'self_conf': 2.612209555214479,
            'swarm_conf': 2.7545198154000348,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'roulette_wheel'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=15) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Constriction Swarm Metaheuristic (CSM) is a population-based optimization algorithm that combines elements of particle swarm optimization with a constriction factor to ensure convergence. The chosen parameters for the 'swarm_dynamic' operator are designed to balance exploration and exploitation, while the use of a uniform distribution helps in maintaining diversity within the population. The results from running the metaheuristic 30 times provide insights into its effectiveness and robustness on the Ackley15 benchmark function.