 # Name: GravitationalSearchAlgorithmMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
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
# This metaheuristic is designed based on the Gravitational Search Algorithm (GSA), which mimics the gravitational force between masses to search for optimal solutions. The algorithm starts with a population of potential solutions distributed randomly in the search space. 
# In this implementation, we use two main operators:
# 1. **Gravitational Search**: This operator uses gravity as the driving force, where the "gravity" parameter represents the strength of the gravitational constant, controlling the range and diversity of the search. The "alpha" parameter controls the exponential decay rate in the algorithm. We set these parameters to default values found in literature for exploration (gravity) and exploitation (alpha).
# 2. **Random Flight**: Inspired by random walks in nature, this operator uses a probabilistic approach where solutions can move towards better regions or explore new areas based on predefined distributions ('levy' is chosen for its long-tail property that aids in exploring remote search spaces). The "scale" parameter adjusts the step size, while "beta" influences the direction of movement. We set these parameters to encourage both exploration and exploitation during the search process.
# Both operators are configured with probabilistic selection ('probabilistic') to ensure a balance between exploitation (trying out new solutions) and exploration (improving upon current best solutions). This combination aims to leverage the strengths of gravitational forces for convergence and random walks for global exploration, making it suitable for optimizing complex multi-modal functions.
# The metaheuristic is initialized with the Sphere benchmark function as our test problem, which is a common choice in optimization due to its simplicity and convexity, ensuring a single global optimum. We set the number of iterations (num_iterations) to 100, allowing sufficient time for convergence while preventing excessive computation.