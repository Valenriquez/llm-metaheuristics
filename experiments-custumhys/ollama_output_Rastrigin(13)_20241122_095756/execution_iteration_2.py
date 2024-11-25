# Name: Spiral Dynamic Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.4764812039295751,
            'angle': 13.468292663819154,
            'sigma': 1.7508863462850675
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the Spiral Dynamic operator to guide the search towards the optimum and a Random Sample operator to ensure exploration. The spiral dynamic operator is particularly useful for problems with complex landscapes, as it can efficiently navigate through them by exploiting the spiral structure of the search space. The random sample operator helps in maintaining diversity and escaping local optima. This combination ensures both exploitation and exploration, leading to effective optimization.

# Error Fixes:
# The provided feedback indicates a SyntaxError due to an invalid character '’' (U+2019). This is likely a result of copying and pasting code from a non-standard source that uses smart quotes instead of straight quotes. To fix this error, replace all occurrences of smart quotes with standard straight quotes in the code.
# Here's the corrected line:
# ```python
# project_dir = Path(__file__).resolve().parents[2]
# ```
# Ensure that all other lines are also using straight quotes throughout your script.