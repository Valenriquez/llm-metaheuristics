 # Name: GravitationalSearchOptimization
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
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

# Short explanation and justification:
# The metaheuristic chosen is Gravitational Search Optimization (GSO). GSO mimics the gravitational force in physics to search for optimal solutions by iteratively adjusting particle positions based on their masses and gravitational forces. In this implementation, two main operators are used: gravitational_search and random_flight. 
# Gravitational search operator uses parameters gravity (1.0) and alpha (0.02), which control the strength of the gravitational force and its influence on the search space, respectively. The selector for this operator is set to 'metropolis', indicating that the selection process follows a probabilistic approach akin to the Metropolis algorithm.
# Random flight operator uses parameters scale (1.0), distribution ('levy'), and beta (1.5). Scale defines the amplitude of random movements, while distribution specifies the type of movement generated—in this case, a levy distribution which is often used in nature-inspired algorithms for its ability to explore diverse regions of the search space. The selector for this operator is 'probabilistic', ensuring that moves are made randomly with a probability weighted by the fitness values.
# These operators together aim to balance between local exploitation and global exploration, making GSO suitable for continuous optimization problems like the Rastrigin function used here.