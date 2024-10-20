 # Name: GravitationalSearchMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(22)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Metaheuristic is designed to mimic the gravitational force between masses in a multi-objective optimization problem. 
# In this case, we use the gravitational search algorithm with specified parameters for gravity (1.0) and alpha (0.02).
# The selector 'all' ensures that all candidate solutions are influenced by the gravitational forces during each iteration, promoting exploration of the solution space.
# This setup is appropriate for a diverse range of optimization problems where global exploration is crucial to avoid local minima.