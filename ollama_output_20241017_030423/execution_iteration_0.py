 # Name: Custom Metaheuristic with Multiple Operators
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Gravitational Search
            'gravitational_search',
            {
                'gravity': 0.8,  # Adjusted gravity for exploration vs exploitation balance
                'alpha': 0.05   # Learning rate for gravitational force calculation
            },
            'metropolis'  # Selector that balances between random and informed search
            ),
            (  
            # Search operator 2: Local Random Walk with Gaussian Distribution
            'local_random_walk',
            {
                'probability': 0.5,  # Probability of performing a local walk
                'scale': 0.75,       # Scaling factor for movement in the search space
                'distribution': 'gaussian'  # Distribution type for random movements
            },
            'probabilistic'  # Selector that favors more probable moves based on energy levels
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is designed with two primary search operators aimed at exploring different aspects of the solution space. 
# The first operator, Gravitational Search, uses a custom gravity parameter to balance the exploration (higher values) versus exploitation (lower values). This encourages both global and local explorations by dynamically adjusting the influence of gravitational forces on particles. The alpha parameter controls the scaling factor for these forces, influencing how quickly solutions converge or diversify. The selector 'metropolis' is used to ensure that some random elements are introduced while maintaining a balance between exploration and exploitation.
# 
# The second operator, Local Random Walk with Gaussian Distribution, introduces stochasticity into local movements within the solution space. By setting the probability of performing such walks to 0.5 and using a Gaussian distribution for the movement scale, this operator encourages both random jumps and gradual adjustments in the search direction. This is particularly useful for escaping local minima by allowing occasional large steps based on a normal (Gaussian) distribution, while maintaining a tendency towards more probable moves that are less likely to stray far from current optimal solutions. The selector 'probabilistic' ensures that such movements are weighted according to their energy levels or likelihood of leading to better fitness values.
# 
# Both operators contribute to the diversity and adaptability required by metaheuristics, with the choice of selectors guiding how aggressively these moves are pursued during each iteration based on the balance between exploration and exploitation desired for the specific problem at hand.