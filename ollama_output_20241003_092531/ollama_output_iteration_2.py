 # Name: GravitationalSearchOptimizer
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Optimizer (GSO) is a metaheuristic inspired by the laws of gravity and mass interaction. 
# In this implementation, we use a single gravitational search operator with parameters 'gravity' set to 1.0 and 'alpha' to 0.02. 
# The selector 'all' means that all particles will follow this search pattern during optimization. 
# We chose GSO because it is effective in handling continuous optimization problems like the Sphere function, which we are using here as a benchmark.
# The parameters for gravity and alpha are set according to typical settings found in literature, where 'gravity' controls the strength of gravitational force and 'alpha' influences the scaling factor in mass calculation.
# This setup ensures that the algorithm explores the search space more effectively by simulating physical laws, which is suitable for a wide range of optimization problems including continuous ones like our benchmark function.