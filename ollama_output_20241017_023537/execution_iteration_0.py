 # Name: MyCustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Random Flight with Gaussian Distribution and Metropolis Selector
                'random_flight',
                {
                    'scale': 0.5,
                    'distribution': 'gaussian',
                    'beta': 1.2
                },
                'metropolis'
            ),
            ( # Search operator 2: Local Random Walk with Uniform Distribution and Probabilistic Selector
                'local_random_walk',
                {
                    'probability': 0.8,
                    'scale': 0.75,
                    'distribution': 'uniform'
                },
                'probabilistic'
            )
        ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve the Rastrigin function optimization problem using two custom operators: Random Flight and Local Random Walk. These operators are chosen based on their flexibility in handling different distributions (Gaussian for Random Flight and Uniform for Local Random Walk) and their application through specific selectors (Metropolis for Random Flight and Probabilistic for Local Random Walk).
# The parameters scale, distribution, beta, probability, and scale are set according to typical values observed in similar metaheuristic implementations. These settings aim to balance exploration and exploitation effectively. For instance, the Gaussian distribution used in the Random Flight operator encourages broader exploration by allowing larger step sizes, while the Uniform distribution in Local Random Walk ensures a more localized search that is still influenced by probabilistic selection rules.
# The Metropolis selector for Random Flight allows it to make use of both greedy and probabilistic approaches depending on the context, which can be beneficial in navigating complex landscapes. Similarly, the Probabilistic selector for Local Random Walk ensures that moves are made with a probability proportional to their expected improvement, enhancing convergence towards better solutions without getting stuck in local minima.
# The choice of num_iterations is set to 200, which provides ample opportunity for the algorithm to refine its search across multiple iterations while preventing excessive computational overhead.
# This setup ensures a balance between explorative and exploitative behaviors that should lead to an effective optimization of the Rastrigin function, as demonstrated by the use of both tailored operators with carefully chosen parameters and selectors.