# Name: Swarm-Based Metaheuristic with Inertial Swarm Optimization

# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    (  # Inertial Swarm Optimization
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses the swarm_dynamic operator from parameters_to_take.txt,
# with the recommended parameters. Inertial swarm optimization is known to be
# effective for continuous optimization problems like the Rastrigin function.
# The greedy selector is used with this operator to select the best-performing
# candidate solution in each iteration.
