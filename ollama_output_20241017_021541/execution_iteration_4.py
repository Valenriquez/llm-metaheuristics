 # Name: GravitationalSearchOptimization
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
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
# The Gravitational Search Optimization (GSO) is a metaheuristic inspired by the gravitational force in physics, where each particle represents a potential solution to the problem. Here, we use the 'gravitational_search' operator with parameters gravity set to 1.0 and alpha to 0.02. This operator will be applied to all particles in the population. Additionally, we include the 'random_flight' operator which uses a scale of 1.0 and distribution type 'levy', with beta set to 1.5. The selector for 'random_flight' is set to 'probabilistic' to ensure that it operates based on probabilistic criteria during each iteration. This combination aims to balance exploration (through gravitational search) and exploitation (through random flight), which should lead to a more effective exploration of the solution space, especially in complex optimization problems like Rastrigin function for dimensions 2.