 # Name: Adaptive Multi-Strategy Metaheuristic (AMSH)
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{fun_name}
prob = fun.get_formatted_problem()

# Define the heuristic operators and their parameters
heur = [
    (
        'genetic_mutation',
        {
            'parameters': {
                'scale': 1.0,
                'elite_rate': 0.1,
                'mutation_rate': 0.25,
                'distribution': "gaussian"
            }
        },
        'metropolis'
    ),
    (
        'random_flight',
        {
            'parameters': {
                'scale': 1.0,
                'distribution': "levy",
                'beta': 1.5
            }
        },
        'probabilistic'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# The Adaptive Multi-Strategy Metaheuristic (AMSH) is designed to explore diverse search strategies, including genetic mutation and random flight, depending on the problem at hand. 
# Genetic mutation with Gaussian distribution helps in exploring different areas of the solution space by introducing small variations, while random flight mimics a foraging behavior that jumps across the search space following a Levy distribution.
# These two operators are chosen because they balance local exploration (through genetic mutation) and global exploration (via random flights), which is crucial for metaheuristics to avoid getting trapped in suboptimal solutions. The parameters for these operators are set according to typical values found in literature, ensuring effective yet diverse search behavior.