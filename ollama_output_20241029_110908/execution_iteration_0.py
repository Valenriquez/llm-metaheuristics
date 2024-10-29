 # Name: GravitationalSearchWithMetropolisSelector
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
# The metaheuristic GravitationalSearchWithMetropolisSelector is designed to solve optimization problems using a combination of gravitational search and metropolis selector operators. 
# The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, which are parameters that control the strength of the gravitational force and the probability of accepting worse solutions during the search, respectively. 
# The random flight operator uses a scale factor of 1.0 and a distribution type of levy, along with beta set to 1.5. This helps in exploring the solution space by allowing for both directed (through gravity) and random movements. 
# Both operators are paired with a metropolis selector, which is chosen based on its ability to balance exploration and exploitation. The gravitational search uses the metropolis selector to ensure that it can jump out of local minima, while the probabilistic nature of the selector complements the explorative capabilities of the levy distribution in the random flight operator. 
# This combination aims to leverage the strengths of both operators to efficiently navigate the complex landscapes typical of many optimization problems, ensuring a balance between exploration and exploitation as required by the problem at hand.