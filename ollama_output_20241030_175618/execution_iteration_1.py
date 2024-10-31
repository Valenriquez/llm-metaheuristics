 # Name: GravitationalSearchAlgorithm
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
# The GravitationalSearchAlgorithm is implemented with two operators: gravitational search and random flight. 
# Gravitational search uses gravity as 1.0 and alpha as 0.02, while the random flight operator scales at 1.0 with a levy distribution and beta of 1.5. 
# Both operators use 'all' as their selector for comprehensive exploration. This setup aims to balance between local exploitation and global exploration through different probabilistic strategies. The GravitationalSearchAlgorithm is designed to optimize problems using metaheuristic techniques, leveraging the strengths of both search methods to find optimal solutions efficiently.