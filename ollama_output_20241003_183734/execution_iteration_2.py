 # Name: GravitationalSearchAlgorithm
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
        'gravity': 0.9,
        'alpha': 0.05
    },
    'metropolis'
    ),
    (  
    'random_flight',
    {
        'scale': 0.8,
        'distribution': 'levy',
        'beta': 1.2
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is chosen for its ability to simulate the gravitational force between particles in a system, which helps in exploring the solution space effectively. In this case, we set the gravity parameter to 0.9 and alpha to 0.05. For the random flight operator, the scale is adjusted to 0.8, with a distribution type of levy and beta value of 1.2. The selector for the gravitational search is set to 'metropolis', which allows it to operate in a probabilistic manner during the exploration phase. This combination helps balance between exploitation and exploration, ensuring that the algorithm can converge to an optimal solution while avoiding premature convergence.
