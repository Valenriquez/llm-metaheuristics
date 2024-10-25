 # Name: GravitationalSearchAlgorithmWithMetropolisSelector
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
        'metropolis'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic using the Gravitational Search Algorithm (GSA) with the Metropolis selector. The GSA is initialized with gravity set to 1.0 and alpha to 0.02, which are parameters for controlling the strength of gravitational force. The selector 'metropolis' is used to decide whether to accept new solutions based on their fitness values, which is appropriate given that we want to explore the search space probabilistically.
# We chose GSA because it mimics the behavior of particles under gravity and can effectively escape local minima by allowing random jumps (similar to a Metropolis selector for accepting new states). The use of 'metropolis' ensures that not all moves are accepted, which is beneficial in avoiding premature convergence to a suboptimal solution.