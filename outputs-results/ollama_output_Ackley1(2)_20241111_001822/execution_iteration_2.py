# Name: ackley_meta
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'mutation_rate': 0.1
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 
# I chose to use the 'local_random_walk' operator first because it has a simple structure and is relatively easy to implement. Then, I used the 'genetic_mutation' operator to introduce some genetic drift and diversity into the population. This should help the metaheuristic escape local optima and find better solutions.
# The choice of parameters for these operators was somewhat arbitrary, but based on my understanding of how they work. For example, the probability in the 'local_random_walk' operator seems too high, and I would expect this to cause the algorithm to overshoot the optimal solution more often than not. Adjusting this value might improve the overall performance of the metaheuristic.
# The use of a 'greedy' selector means that the fittest individual is selected at each step, which can lead to premature convergence if not carefully balanced with the randomness introduced by the operators. This may need to be adjusted depending on the specific problem being tackled.