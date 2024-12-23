# Name: Hybrid Evolutionary Algorithm with Local Search
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
    (  # Search operator 1: Evolutionary Algorithm (EA)
        'evolution',
        {
            'population_size': 50,
            'mutation_rate': 0.2,
            'crossover_rate': 0.7
        },
        'tournament_selection'
    ),
    (
        # Local search operator 1: Hill Climbing
        'hill_climb',
        {
            'max_iterations': 10,
            'step_size': 0.01
        },
        'best_first_search'
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
# This hybrid algorithm combines Evolutionary Algorithm (EA) with Local Search (Hill Climbing). The EA is used to explore the search space globally by generating new solutions through mutation and crossover. The local search is then applied to refine these solutions locally, ensuring that they are as close as possible to an optimal solution. By combining both strategies, the algorithm aims to efficiently balance exploration and exploitation, leading to better performance on the Rastrigin function.
```

This response provides a metaheuristic algorithm named "Hybrid Evolutionary Algorithm with Local Search" using the provided operators and parameters. The algorithm combines Evolutionary Algorithm (EA) with Local Search (Hill Climbing) to explore and refine solutions for the given benchmark function (Rastrigin in this case).