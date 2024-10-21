 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin
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
# The Gravitational Search algorithm is designed to mimic the gravitational force between masses, 
# where each mass represents a potential solution in the search space. In this case, we have set the gravity constant to 1.0, which influences the strength of the gravitational pull among solutions. Alpha (α) is set to 0.02, affecting the acceleration of the convergence towards better solutions.
# The Gravitational Search algorithm uses two operators: 'gravitational_search' and 'random_flight'. The 'gravitational_search' operator utilizes both gravity for a directed search and random forces for exploration. It includes parameters 'gravity' set to 1.0, which represents the strength of the gravitational constant, and 'alpha' set to 0.02, controlling the scaling factor for the convergence towards better solutions.
# The second operator is 'random_flight', designed to introduce diversity in the search by using a random flight mechanism. Parameters include 'scale' at 1.0 which sets the scale of the random jumps, and 'distribution' set to 'levy'. This distribution type helps in exploring regions that might not be reached by other methods, providing an advantage for escaping local minima. The selector for this operator is 'probabilistic', indicating its probabilistic nature, suitable for cases requiring a balance between exploration and exploitation.
# Both operators are used in conjunction with the 'all' selector to ensure they both contribute their strengths across all iterations of the algorithm, enhancing overall performance and convergence speed. This setup leverages the benefits of both gravitational attraction for convergence and random exploration for avoiding local minima, making it well-suited for complex optimization tasks like those encountered in benchmark functions such as Rastrigin's function.