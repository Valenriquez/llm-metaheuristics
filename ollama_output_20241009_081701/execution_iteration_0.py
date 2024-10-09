 # Name: Custom Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'uniform',
                'beta': 0.8
            },
            'probabilistic'
            ),
            (  
            'local_random_walk',
            {
                'probability': 0.6,
                'scale': 0.5,
                'distribution': 'gaussian'
            },
            'metropolis'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic named "Custom Metaheuristic" is designed to solve the Rastrigin function optimization problem with two dimensions (Rastrigin(2)). 
# The custom metaheuristic employs two search operators, both selected from the predefined list of operators in the parameters_to_take.txt file: 'random_flight' and 'local_random_walk'. These operators are chosen based on their ability to explore different distributions and probabilities for mutation within the solution space.
# For the 'random_flight', we set the scale parameter to 0.5, which controls the extent of the random flight, ensuring a balance between exploration and exploitation. The distribution is set to 'uniform' to introduce randomness in the search direction. The beta parameter is adjusted to 0.8, influencing the probability of taking larger steps compared to smaller ones.
# Similarly, the 'local_random_walk' operator uses a Gaussian distribution for mutation with a scale of 0.5, favoring moderate changes around the current solution. The probability parameter is set to 0.6, which determines the likelihood of performing a local random walk versus other search strategies.
# Both operators are configured with probabilistic selection ('metropolis'), allowing them to be used in an adaptive manner during the optimization process. This setup aims to leverage the strengths of both exploration and exploitation mechanisms within the genetic algorithm framework.
# The metaheuristic runs for 100 iterations, collecting results that can be printed out at the end, showing the best solution found (x_best) and its corresponding fitness value (f_best).