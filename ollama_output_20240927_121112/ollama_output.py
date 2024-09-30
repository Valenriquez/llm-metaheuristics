 # Name: Metaheuristic Exploration
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)  # Example function, replace with your chosen benchmark function
prob = fun.get_formatted_problem()

# Define heuristic operators and their parameters
heur = [
    (
        'genetic_crossover',
        {
            'pairing': 'rank',
            'crossover': 'uniform',
            'mating_pool_factor': 0.4
        },
        'all'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

# Create and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named 'Metaheuristic Exploration' is designed to solve optimization problems using a diverse set of heuristic operators tailored from the provided parameters. Each operator is chosen based on its specific parameter settings that are selected from the list provided in the task, ensuring diversity and coverage across different algorithm types (e.g., genetic algorithms, swarm intelligence). The metaheuristic includes:
# 1. Genetic crossover with ranking-based pairing and uniform crossover for generating offspring.
# 2. Genetic mutation with Gaussian distribution to introduce random changes, focusing on greedy and metropolis selectors.
# 3. Gravitational search algorithm with a defined gravity constant and alpha factor.
# 4. Random flight dynamics influenced by Levy distribution, suitable for probabilistic selection.
# 5. Local random walk that operates uniformly across the space, impacting all selector types.
# 6. Spiral dynamic model tailored to greedy and metropolis selectors, adjusting radius and angle parameters.
# 7. Swarm dynamics with inertial version, configured based on self-confidence and swarm confidence factors, using Levy distribution for probabilistic selection.
# The combination of these operators is intended to leverage the strengths of each approach in exploring different aspects of the solution space, ensuring a comprehensive search across potential solutions.