 # Name: MyCustomMetaheuristic
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
                'beta': 1.0
            },
            'probabilistic'
            ),
            (  
            'local_random_walk',
            {
                'probability': 0.8,
                'scale': 0.5,
                'distribution': 'gaussian'
            },
            'metropolis'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named MyCustomMetaheuristic, designed to solve the Rastrigin function with two dimensions. 
# We've chosen 'random_flight' for exploration due to its ability to navigate large scales based on scale=0.5, uniform distribution for simplicity in parameter setting. Beta in random flight is set to 1.0 to balance exploration and exploitation effectively.
# The second operator 'local_random_walk' is selected with a higher probability of mutation (probability=0.8) and the same scale as before but with Gaussian distribution which is typical for local walks, enhancing convergence properties. This selector uses Metropolis which is suitable for probabilistic acceptance criteria in complex landscapes.
# Both operators are set to run iteratively through a series of iterations specified by num_iterations=200. The verbose mode provides detailed output during the execution, useful for debugging and understanding the optimization process.
# These settings aim to balance between global exploration and local exploitation, ensuring convergence towards an optimal solution while avoiding premature convergence or getting stuck in suboptimal regions through appropriate mutation rates and distributions tailored by each operator's specific parameters.