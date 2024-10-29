 # Name: GravitationalSearchWithMetropolisSelector
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
    'metropolis'
    ),
    (  
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'probabilistic'
)
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is Gravitational Search with Metropolis Selector for optimization of the Rastrigin function in a 2-dimensional space. 
# Gravitational search mimics the gravitational force among particles to explore the solution space, while the metropolis selector uses probabilistic criteria to decide whether to accept new solutions or not, which helps in escaping local minima.
# The Rastrigin function is chosen for its properties as a standard benchmark for testing optimization algorithms due to its many local minima, making it suitable for evaluating the ability of metaheuristics to find good solutions globally.
# Gravitational search parameters include gravity and alpha which control the strength and range of gravitational forces applied on particles respectively.
# Local random walk operator is configured with probability set high (0.75) to encourage exploration away from current positions, and distribution uniform for even spread across the solution space.
# Selector 'metropolis' allows the algorithm to occasionally accept worse solutions in its search process, enhancing its ability to avoid local minima.
