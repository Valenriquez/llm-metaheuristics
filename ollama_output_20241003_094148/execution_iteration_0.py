 # Name: Adaptive Random Search (ARS) Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Random Flight
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'uniform',
                'beta': 2.0
            },
            'probabilistic'
            ),
            (  
            # Search operator 2: Local Random Walk
            'local_random_walk',
            {
                'probability': 0.8,
                'scale': 0.75,
                'distribution': 'gaussian'
            },
            'probabilistic'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Adaptive Random Search (ARS) metaheuristic is designed to explore solutions in a problem space through diverse search operators. In this implementation, we use two main operators: 'random_flight' and 'local_random_walk'. These operators are selected based on their probabilistic nature ('probabilistic' selector), which allows them to balance exploration and exploitation during the optimization process.
# The 'random_flight' operator is configured with a scale of 0.5, using a uniform distribution for its movement, and a beta value of 2.0 to control the step size. This helps in both global exploration and fine-tuning around current best solutions.
# The 'local_random_walk' operator operates with an increased probability of 0.8, which encourages more frequent local search movements. It uses Gaussian distribution for its movement scaling, allowing for more focused adjustments near the current solution. Both operators are configured to work in conjunction, leveraging their distinct characteristics to effectively navigate the optimization landscape.
# The metaheuristic is set to run for 200 iterations, providing ample opportunity for the population of solutions to adapt and converge towards an optimal or near-optimal configuration. This approach aims to balance between global search capabilities and local refinement through carefully chosen parameter settings and operator selection based on the problem requirements as specified in parameters_to_take.txt.