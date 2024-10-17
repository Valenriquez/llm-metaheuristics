 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
    ),
    ( # Random Flight Operator
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
# This metaheuristic combines two operators from the provided list: Gravitational Search and Random Flight. 
# The Gravitational Search is designed to mimic the gravitational forces between particles, which aids in exploring the solution space. It uses parameters such as gravity and alpha to control its behavior. In this case, both are set at their default values recommended for general use.
# The Random Flight operator introduces a random component into the search process, allowing for exploration of new areas beyond local minima. Parameters include scale, which controls the amplitude of the random jumps, and distribution, determining whether these jumps follow a Levy, uniform, or Gaussian distribution. Here, the Levy distribution is selected, known to effectively escape local minima during optimization tasks. The beta parameter in this case follows typical settings for such distributions.
# Both operators are configured with 'all' as their selector, meaning they will be applied on every iteration of the metaheuristic algorithm. This combination aims to leverage both gravitational attraction towards high-quality solutions and random exploration to efficiently converge to an optimal solution across a range of potential functions, including the Rastrigin function used in this example.