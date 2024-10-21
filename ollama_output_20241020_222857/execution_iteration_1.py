# Name: Hybrid Metaheuristic with Gravitational Search and Random Walk

# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)   
prob = fun.get_formatted_problem()

heur = [
    ('gravitational_search', {'gravity': 1.0, 'alpha': 0.02}, 'greedy'),
    ('local_random_walk', {'probability': 0.8, 'sigma': 0.1}, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines gravitational search and local random walk operators.
# Gravitational search explores the search space by simulating the gravitational pull on particles.
# Local random walk explores the neighborhood of the current solution by making small random jumps.
# The combination of these operators can help the metaheuristic find good solutions in the Rastrigin(2) problem.