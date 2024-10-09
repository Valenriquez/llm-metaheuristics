 # Name: GravitationalSearchOptimization
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
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
# The Gravitational Search Optimization (GSO) algorithm is a metaheuristic inspired by the principles of gravitational force and mass interaction. 
# In this implementation, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02.
# The selector used is 'all', meaning that all particles will follow this search strategy. This approach aims to mimic the gravitational forces between masses in a multidimensional space, allowing for exploration of potential solutions across various regions of the search space.
# GSO is particularly useful for continuous optimization problems like the Rastrigin function here, where it can effectively navigate the landscape by adjusting the 'gravity' parameter which influences the strength of the gravitational pull and the 'alpha' parameter which controls the scaling factor affecting the motion of particles towards better solutions.