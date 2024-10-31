 # Name: GravitationalSearchAlgorithm
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
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the laws of gravity and motion in physics. In this implementation, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector 'all' indicates that these operators will be applied to all particles or solutions during each iteration. This approach is used because GSA can handle exploration of diverse regions in the search space, potentially leading to better global optimization by considering multiple potential solutions simultaneously.
# The Rastrigin function, chosen as the benchmark problem, has a smooth and unimodal landscape, which makes it suitable for testing convergence properties and robustness of different metaheuristic algorithms. The GSA is applied with its specific parameters that define the strength of gravitational forces and the influence of alpha on solution adjustment.
# By setting 'gravity' to 1.0, we emphasize the primary role of gravity in driving the search towards better solutions. The parameter 'alpha' controls the scaling factor for movement based on the gradients; a value of 0.02 encourages small steps that might help in avoiding premature convergence and exploring finer details of the function landscape.
# The selector 'all' ensures that these operators are applied to every element in the population, allowing the algorithm to adapt its search strategy throughout the iterations, potentially leading to an optimal balance between exploration (searching new areas) and exploitation (refining solutions within current knowledge).