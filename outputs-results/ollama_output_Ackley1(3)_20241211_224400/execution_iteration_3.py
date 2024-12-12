# Name: Hybrid Adaptive Evolutionary Metaheuristic (HAEM)

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(4)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'genetic_algorithm',
        {
            'mutation_rate': 0.05,
            'crossover_rate': 0.8,
            'population_size': 50
        },
        'greedy'
    ),
    (
        'simulated_annealing',
        {
            'initial_temperature': 1000,
            'cooling_rate': 0.99,
            'num_iterations': 100
        },
        'probabilistic'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=10) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The HAEM combines the strengths of genetic algorithms and simulated annealing to create a hybrid metaheuristic for optimization problems. Genetic algorithms excel at global search due to their parallel processing capabilities, while simulated annealing is adept at fine-tuning solutions in the neighborhood of local optima. By integrating these two approaches, HAEM can effectively explore large solution spaces while efficiently refining solutions around optima. This makes it suitable for a wide range of optimization problems requiring both exploration and exploitation strategies.