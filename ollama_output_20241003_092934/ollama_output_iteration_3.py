 # Name: PSOwithAdaptiveParameters
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'swarm_dynamic',
    { 
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'gaussian'
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
# The PSOwithAdaptiveParameters metaheuristic uses two main operators, swarm_dynamic and random_flight, to explore the solution space. 
# Swarm dynamics are set with a factor of 0.7, self-confidence of 2.54, swarm confidence of 2.56, and version set to inertial. The distribution is chosen as gaussian for diversity in exploration.
# Random flight operator has scale set to 1.0, distribution as levy, and beta value of 1.5. This operator uses probabilistic selection to balance between exploitation and exploration.
# Both operators are applied with the selector 'all' which means they will be used during every iteration of the metaheuristic. This combination aims to leverage both local search capabilities of swarm dynamics and global search abilities of random flight, adapting parameters as needed based on the problem characteristics from parameters_to_take.txt.