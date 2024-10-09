 # Name: GravitationalSearchMetaheuristic
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
# This code implements a Gravitational Search Metaheuristic for optimizing the Rastrigin function with 2 dimensions. The Gravitational Search algorithm is characterized by two main parameters: gravity (1.0) and alpha (0.02). These parameters are set according to the specifications in parameters_to_take.txt.
# The selector used here is 'metropolis', which suggests that the selection process for candidate solutions follows a probabilistic approach, similar to the Metropolis-Hastings algorithm, ensuring diversity and exploration of the search space while also considering promising regions.
# This setup aims to balance between global exploration and local exploitation by leveraging gravitational forces modeled in physics, where lighter objects are attracted to heavier ones based on their mass ratios, mimicking the behavior of particles moving towards more favorable areas in the search space. The use of 'metropolis' selector enhances the probabilistic nature of this algorithm, allowing it to avoid getting stuck in local minima and explore various parts of the function landscape efficiently.