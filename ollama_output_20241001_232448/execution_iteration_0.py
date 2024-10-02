 # Name: Custom Metaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_mutation',
    {
        'scale': 0.5,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'uniform'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The metaheuristic is named Custom Metaheuristic. It uses a genetic mutation operator with specified parameters for scale, elite rate, mutation rate, and distribution type. The selector used here is probabilistic to determine the application of this mutation across iterations based on predefined probabilities associated with each selection method in the parameters_to_take.txt file. This setup allows for exploration and exploitation trade-offs through controlled random changes within a population of candidate solutions, which is typical in genetic algorithms or other similar metaheuristic approaches where mutations play a central role.
