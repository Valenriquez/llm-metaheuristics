# Name: Hybrid Metaheuristic using Random Search and Differential Mutation

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
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.3213739938633245,
            'distribution': 'levy'
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 0.515275161513281
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
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
# This hybrid metaheuristic combines Random Search and Differential Mutation to explore the search space effectively. The Random Search operator helps in exploring diverse regions of the solution space using a Levy distribution, which can lead to more efficient exploration. The Differential Mutation with 'best' expression ensures that the best solution found so far is used as a template for generating new solutions, promoting convergence towards better solutions. The use of a probabilistic selector allows for a balance between exploitation (using good solutions) and exploration (exploring new regions). This combination is particularly useful for problems where local optima may trap simple search methods, as the hybrid approach can escape these traps and potentially find better solutions.