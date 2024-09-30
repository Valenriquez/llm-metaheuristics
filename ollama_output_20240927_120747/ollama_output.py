 # Name: MetaHeuristicX
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere()  # Replace with the actual function name if different
prob = fun.get_formatted_problem()

heur = [(
    'genetic_crossover',
    {
        'pairing': 'rank',
        'crossover': 'blend',
        'mating_pool_factor': 0.4
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# This metaheuristic is named MetaHeuristicX and uses the Sphere function as the benchmark problem. The genetic crossover operator with ranking pairing, blend crossover type, and a mating pool factor of 0.4 are employed. All selectors (search strategies) are utilized for exploration. The heuristic configuration is based on the provided parameters from parameters_to_take.txt, ensuring that each component of the algorithm is tailored according to the specifications. This setup allows for a comprehensive search across multiple operators and parameter settings, providing robust results in solving optimization problems.