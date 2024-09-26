 # Name: Metaheuristic Explorer
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)  # Example function, replace with the actual function name
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'genetic_crossover',
    {
        'pairing': 'rank',
        'crossover': 'two',
        'mating_pool_factor': 0.4
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic named "Metaheuristic Explorer" using the provided operators and their parameters. It starts by setting up the necessary path for importing modules, then it initializes the benchmark function (in this case, Sphere) to be optimized. 
# A genetic crossover operator is selected with specific parameters such as pairing set to 'rank' and crossover type to 'two'. The mating pool factor is set to 0.4. This heuristic is applied using a greedy selector. 
# The metaheuristic is then instantiated, configured to run for 100 iterations, and the results are printed out showing the best solution found.