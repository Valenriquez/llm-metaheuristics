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
# This code implements the Gravitational Search Algorithm (GSA) with a Metropolis selector for optimization of the Rastrigin function in 2 dimensions. The GSA is designed to mimic gravitational interactions among particles, where each particle represents a candidate solution. The algorithm uses two parameters: gravity (1.0) and alpha (0.02), which control the strength of the gravitational force and the scaling factor for velocity updates, respectively. The selector 'metropolis' is chosen because it allows for probabilistic transitions between solutions, which can help in exploring diverse regions of the search space more effectively than a purely greedy approach. This combination aims to balance exploration and exploitation, ensuring that the algorithm converges towards an optimal solution while avoiding local minima.
