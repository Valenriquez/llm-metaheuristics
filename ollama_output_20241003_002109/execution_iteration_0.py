 # Name: MyCustomMetaheuristic
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
                'scale': 0.8,
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
# The code defines a custom metaheuristic named MyCustomMetaheuristic, which uses two search operators: Random Flight and Local Random Walk. 
# These operators are chosen based on the parameters provided in the parameters_to_take.txt file, ensuring that only allowed operators and their specific parameters are utilized. 
# The Random Flight operator is configured with a scale of 0.5, distribution set to 'uniform', and beta of 2.0. This operator uses probabilistic selection as its selector.
# Similarly, the Local Random Walk operator has a probability of 0.8, a scale of 0.8, and a distribution type of 'gaussian'. It also employs probabilistic selection.
# Both operators are configured to run for 200 iterations, providing ample opportunity for the algorithm to explore the solution space. The verbose mode is enabled to print detailed information about each iteration's progress.
# This setup aims to leverage the strengths of both operators—Random Flight for global exploration and Local Random Walk for local fine-tuning—to find an optimal solution efficiently.