 # Name: MyCustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
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
            'local_random_walk', # Search operator 2: Local Random Walk
            {
                'probability': 0.8,
                'scale': 1.5,
                'distribution': 'gaussian'
            },
            'metropolis'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve the Rastrigin function with two operators, each tailored for different search dynamics.
# 1. **Random Flight**: This operator uses a uniform distribution for its scale parameter, which aids in exploring diverse regions of the search space according to the problem's characteristics as inferred from parameters_to_take.txt. The beta value is set at 2.0 to influence how aggressively this exploration occurs.
#    - **Parameter Justification**: The scale parameter is adjusted down from the default value (1.0) to 0.5, which is expected to facilitate more frequent and varied movements within the search space during each iteration. This approach balances between exploitation and exploration as suggested by the parameters_to_take.txt.
#    - **Selector Justification**: 'probabilistic' selector is chosen because it aligns with the probabilistic nature of random flight, allowing for a chance-driven movement through the solution space.
# 2. **Local Random Walk**: This operator employs a Gaussian distribution to adjust its steps based on current conditions in the search space. The scale parameter is increased (to 1.5) compared to the default value, which encourages more dynamic local search capabilities as per parameters_to_take.txt recommendations.
#    - **Parameter Justification**: A higher probability of mutation (0.8) ensures that the search focuses on improving solutions in promising areas by perturbing them. The Gaussian distribution for this operator is appropriate to mimic natural random walks but weighted towards more likely improvements closer to current optima found during the search process.
#    - **Selector Justification**: 'metropolis' selector is selected because it uses a probabilistic approach similar to the local random walk, allowing for acceptance of less optimal solutions based on predefined criteria that might lead to an improvement in fitness after perturbation.
# The combination of these two operators with tailored parameters and selection strategies aims to optimize the Rastrigin function effectively by adapting search patterns dynamically during multiple iterations, as specified by num_iterations=200. This setup is expected to yield better convergence towards the global optimum or a near-optimal solution compared to using default settings from the framework.