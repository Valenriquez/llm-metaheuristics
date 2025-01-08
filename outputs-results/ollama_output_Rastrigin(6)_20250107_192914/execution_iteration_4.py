# Name: Random Sample and Local Walk Metaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The metaheuristic uses two operators: 'random_sample' and 'local_random_walk'.
# 'random_sample' is used to generate initial solutions randomly.
# 'local_random_walk' is applied with a probability of 0.75, scale of 1.0, and uniform distribution to explore the neighborhood around each solution.
# The selector 'greedy' ensures that if a better solution is found, it will be accepted immediately.
# The selector 'metropolis' allows for some acceptance of worse solutions based on a probability, aiding in avoiding local optima.

# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin(6)_20250107_192914/execution_iteration_4.py", line 58, in <module>
#     met.run()
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 144, in run
#     self.apply_search_operator(perturbator, selector)
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 100, in apply_search_operator
#     exec('Operators.' + operator_name + '(self.pop,' + operator_params)
#   File "<string>", line 1, in <module>
# TypeError: random_sample() got an unexpected keyword argument 'parameter1'
#
# This error occurs because the 'random_sample' operator does not accept any parameters.
# The fix is to remove the empty dictionary for 'parameter1' in the heur list.