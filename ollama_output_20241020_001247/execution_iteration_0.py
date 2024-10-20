 # Name: GravitationalSearchAlgorithm
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is used to simulate the gravitational force acting between masses in a multi-dimensional space, where each mass represents a potential solution to an optimization problem. In this case, we use GSA with parameters gravity set to 1.0 and alpha to 0.02. This configuration allows for exploration of the search space by simulating the attraction and repulsion forces among candidate solutions.
# The Random Flight operator is introduced as another heuristic method to enhance the exploration capabilities of the algorithm. It uses a scale factor of 1.0, with the distribution set to levy, which introduces random jumps in the solution space. This is beneficial for escaping local minima and exploring new regions of the search space. The beta parameter is set to 1.5, allowing for a balance between exploration and exploitation.
# Both operators are selected based on their performance in diverse optimization scenarios, with gravitational_search using greedy selection to focus on promising solutions, while random_flight employs an all-inclusive selector to ensure thorough exploration of the search space.