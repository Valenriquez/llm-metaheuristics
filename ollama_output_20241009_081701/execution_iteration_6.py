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
# This metaheuristic combines the Gravitational Search Algorithm (GSA) with a Random Flight mechanism to explore the search space. GSA is designed to mimic the gravitational force among particles in a multi-dimensional space, where each particle represents a potential solution. The 'gravity' parameter controls the strength of the gravitational attraction, while 'alpha' adjusts the step size. The Gravitational Search operator uses these parameters to iteratively adjust the positions of particles based on their masses and distances.
# Random Flight is introduced to enhance exploration by allowing the algorithm to take random steps in the search space, which can help escape local minima. Parameters include 'scale', determining the magnitude of these steps, and 'distribution' that defines whether these steps are uniformly distributed ('uniform') or follow a Levy flight pattern for enhanced exploration. The 'beta' parameter influences the nature of the distribution used.
# Both operators utilize probabilistic selection criteria ('probabilistic') to decide when to apply their respective mechanisms, ensuring a balance between exploitation and exploration during optimization. This hybrid approach aims to leverage the strengths of both gravitational attraction (for convergence) and random jumps (for diversity), providing a robust search mechanism for global optimization problems.