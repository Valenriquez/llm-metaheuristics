 # Name: MyCustomMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)  # Example function, replace with your chosen benchmark function
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'genetic_crossover',
    {
        'pairing': 'random',
        'crossover': 'blend',
        'mating_pool_factor': 0.4
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve optimization problems using a combination of genetic operators and probabilistic selection. 
# We start by selecting the Sphere function as our benchmark problem, which typically represents an n-dimensional space where the goal is to find the minimum value.
# Next, we configure the heuristic with a single operator from the genetic suite: 'genetic_crossover'. This crossover operation involves random pairing ('random'), blending of genetic materials for offspring creation ('blend'). 
# The mating pool factor is set at 0.4, which influences the size of the subset of individuals used in reproduction.
# We select a probabilistic selector that considers all, greedy, metropolis, and probabilistic approaches to ensure diversity and exploration across different strategies within our heuristic framework.
# This setup aims to balance exploitation (using blended genetics) with exploration (probabilistic selection), allowing the algorithm to efficiently converge towards an optimal solution while avoiding local minima.