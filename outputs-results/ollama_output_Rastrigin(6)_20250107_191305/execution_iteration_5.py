# Name: Hybrid Metaheuristic Algorithm
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.5,
            'beta': 1.5
        },
        'probabilistic'
    ),
    (
        'differential_evolution',
        {
            'factors': [0.7, 1.3],
            'crossover_probability': 0.9
        },
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=200, num_agents=150)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic algorithm combines the strengths of three different search operators: Random Search, Central Force Dynamics, and Differential Evolution. The Random Search operator helps to explore the solution space randomly, while the Central Force Dynamics operator simulates a system of particles interacting with each other based on attractive forces, guiding the search towards better solutions. The Differential Evolution operator is used for local refinement by combining multiple candidate solutions in a population-based manner.
# The metropolis selector is employed for the Random Search and Central Force Dynamics operators to ensure that random movements are accepted probabilistically if they lead to an improvement in the objective function value.
# The differential evolution operator uses a probabilistic approach to generate new candidate solutions, making it efficient for exploring complex landscapes.
# This combination allows for a robust search strategy that balances exploration and exploitation, leading to improved performance on optimization problems.