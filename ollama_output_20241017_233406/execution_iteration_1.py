 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        "gravitational_search",
        { 
            "gravity": 1.0,
            "alpha": 0.02
        },
        "metropolis"
    ),
    (  
        "random_flight",
        {
            "scale": 1.0,
            "distribution": "levy",
            "beta": 1.5
        },
        "probabilistic"
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm with specified parameters for both operators. The first operator is the gravitational search itself, configured with gravity as 1.0 and alpha as 0.02. It uses the metropolis selector to decide whether to accept new solutions based on probability.
# The second operator is a random flight, which scales by 1.0 and follows a levy distribution with beta set to 1.5. This operator also employs probabilistic selection. Both operators are used in conjunction within the gravitational search framework to explore potential solutions for optimization problems like Rastrigin's function.
# The use of both gravitational search and random flight reflects an attempt to balance exploration (random flight) and exploitation (gravitational search), which is typical in metaheuristic algorithms aiming to mimic natural phenomena or processes for solving complex optimization tasks.