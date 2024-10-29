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
                'mutation',
                {
                    'mutation_rate': 0.25,
                    'distribution': 'uniform'
                },
                'metropolis'
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

# Explanation and Justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve optimization problems using a combination of mutation and random flight search operators. 
# The Rastrigin function, which has a dimension of 2, serves as the benchmark problem for this experiment.

# Two heuristic operators are employed in this approach:
# 1. Mutation Operator: This operator introduces randomness into the solution by altering the mutation rate and distribution type according to parameters specified in the parameters_to_take.txt file. The mutation rate is set at 0.25, indicating that each gene has a 25% chance of being mutated during reproduction. The distribution for mutations is uniformly distributed, allowing for equal likelihoods across all possible values within the solution space.
# Justification: Mutation is a fundamental operator in evolutionary algorithms, helping to explore new areas of the search space and avoid local minima by preventing premature convergence. Uniform distribution ensures that changes are spread out evenly, which can be beneficial when no specific information about the problem's landscape is available.

# 2. Random Flight Operator: This operator utilizes a scaling factor (scale) and a specific probability distribution (levy). The scale parameter determines how much the solution should vary during each iteration, while the levy distribution introduces long-range jumps that can help escape from local optima. The beta value is set to 1.5, which influences the shape of the distribution toward more extreme values, potentially aiding in global exploration.
# Justification: Random flight operators are useful for exploring large swaths of the search space when traditional gradient methods might get stuck in local minima. Levy distribution's long tails allow for a higher probability of jumping to distant locations compared to Gaussian or uniform distributions, which is ideal for problems with complex topography and multiple peaks.

# Both operators use 'metropolis' as their selector, indicating that they will operate according to the metropolis decision-making criteria during each iteration. This ensures that moves are accepted based on a probabilistic acceptance rule, allowing potentially beneficial but less optimal moves to be considered, which can help avoid getting stuck in suboptimal solutions.
# Justification: Metropolis selection is appropriate here because it allows for both explorative and exploitative phases within the same algorithm framework by probabilistically choosing between accepting new solutions and keeping the current best found solution. This flexibility is crucial for navigating complex optimization landscapes without becoming overly biased towards one type of solution pathway.