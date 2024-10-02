 # Name: MyCustomMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the heuristic operators with parameters from parameters_to_take.txt
heur = [( # Search operator 1
    'genetic_mutation',
    {
        'parameters': {
            'scale': 0.5,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        }
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The code defines a custom metaheuristic named MyCustomMetaheuristic, which uses the genetic mutation operator with specified parameters. 
# The Rastrigin function is chosen as the benchmark problem, and it operates over two dimensions (n=2).
# The heuristic includes only one search operator, 'genetic_mutation', configured with specific parameter values from the provided template.
# These parameters are selected to balance between exploration and exploitation typical in genetic algorithms, such as a moderate mutation rate and elite preservation for better convergence. 
# The probabilistic selector is used to apply this operator probabilistically during the search process, which can help in exploring various parts of the solution space without getting stuck too early.
# The metaheuristic runs for 100 iterations, and results are printed including the best found solution and its fitness value after optimization completes.