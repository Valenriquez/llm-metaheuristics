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
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    (  
        'random_flight',
        {
            'scale': 1.0,
            'distribution': "levy" or "uniform" or "gaussian",
            'beta': 1.5
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 3
        'local_random_walk',
        { 
            'probability': 0.75,
            'scale': 1.0,
            'distribution': "uniform" or "gaussian" or "levy"
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 4
        'random_sample',
        {},
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 5
        'spiral_dynamic',
        { 
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 6
        'swarm_dynamic',
        { 
            'factor': 0.7 or 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial" or "constriction",
            'distribution': "uniform" or "gaussian" or "levy"
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic named GravitationalSearchMetaheuristic, which utilizes various search operators to solve an optimization problem. It starts by importing necessary modules for the benchmark function and metaheuristic implementation. 
# A Rastrigin function with dimension 2 is selected as the objective function. The heuristic list includes six operators: gravitational_search, random_flight, local_random_walk, random_sample, spiral_dynamic, and swarm_dynamic. These operators are configured with parameters specific to each operator's functionality, ensuring that only the allowed parameters from the provided template are utilized.
# The selectors for these operators are set to 'greedy', 'all', 'metropolis', or 'probabilistic'. This setup allows for a comprehensive exploration of solution space by combining different search strategies with various probabilistic and deterministic mechanisms.
# Each operator is initialized with its corresponding parameters, which are taken from the template provided in the task description. The metaheuristic runs for 100 iterations, and final results display the best found solution's coordinates and function value.