 # Name: Multi-Strategy Evolutionary Algorithm (MSEA)
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators with their respective parameters and selectors
heur = [
    (
        'genetic_mutation',  # Operator name
        {
            'scale': 1.0,  # Parameter for mutation scale
            'elite_rate': 0.1,  # Parameter for elite rate
            'mutation_rate': 0.25,  # Parameter for mutation rate
            'distribution': 'gaussian'  # Parameter for distribution type
        },
        'probabilistic'  # Selector name
    ),
    (
        'random_flight',  # Operator name
        {
            'scale': 1.0,  # Parameter for scale
            'distribution': 'levy',  # Parameter for distribution type
            'beta': 1.5  # Parameter for beta value
        },
        'probabilistic'  # Selector name
    ),
    (
        'spiral_dynamic',  # Operator name
        {
            'radius': 0.9,  # Parameter for radius
            'angle': 22.5,  # Parameter for angle
            'sigma': 0.1  # Parameter for sigma value
        },
        'probabilistic'  # Selector name
    )
]

# Initialize and run the metaheuristic with defined operators and parameters
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The Metaheuristic named Multi-Strategy Evolutionary Algorithm (MSEA) is designed to explore multiple search operators tailored for different probabilistic selection mechanisms. 
# The chosen operators include genetic mutation, random flight, and spiral dynamic search, each with specific parameters that influence their behavior in the optimization process. 
# Genetic mutation introduces variation through a Gaussian distribution affecting both elite rate and mutation scale, while random flight simulates movement influenced by Levy distributions to facilitate exploration. 
# Spiral dynamic search mimics a spiraling pattern which can be useful for navigating complex landscapes. These operators are selected probabilistically to dynamically adjust during the optimization process based on predefined rules and parameters that balance exploitation and exploration. 
# The MSEA leverages these diverse operators in combination with probabilistic selectors, allowing for an adaptive and flexible approach to solving complex optimization problems, as supported by the provided parameter templates from a hypothetical research project context.