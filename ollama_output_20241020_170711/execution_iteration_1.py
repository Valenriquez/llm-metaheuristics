 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
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
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named GravitationalSearchMetaheuristic. It uses the Rastrigin function as the benchmark problem, which has a dimension of 2. Two search operators are employed in this approach: gravitational_search and random_flight. The gravitational_search operator employs parameters gravity (set to 1.0) and alpha (set to 0.02). The selector for both operators is set to 'all'. On the other hand, the random_flight operator uses scale (fixed at 1.0), distribution ('levy'), and beta (set to 1.5). The selector for this operator is 'probabilistic'. This combination aims to explore the solution space with a balance between deterministic and probabilistic search strategies. The gravitational_search leverages gravity and alpha to influence the convergence towards better solutions, while random_flight introduces stochastic elements through its distribution and beta parameters, which can help in escaping local minima and exploring new areas of the solution space. Both operators contribute to the overall performance by combining algorithmic stability with a degree of randomness for avoiding premature convergence.
# The Rastrigin function is chosen because it is a standard benchmark for testing optimization algorithms due to its many local minima, which makes it challenging yet suitable for evaluating the performance and robustness of different metaheuristic approaches.