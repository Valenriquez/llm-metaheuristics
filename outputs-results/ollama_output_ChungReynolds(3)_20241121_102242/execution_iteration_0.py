# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.ChungReynolds(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the swarm_dynamic and spiral_dynamic operators. 
# The swarm_dynamic operator is used to perform collective exploration of the solution space,
# while the spiral_dynamic operator helps in fine-tuning the solutions near the optimum.
# The combination aims to leverage both global and local search capabilities for better convergence.
# The use of the 'probabilistic' selector allows each operator to be executed with a certain probability
# at each iteration, facilitating exploration and exploitation.

# In case there was an error, please fix it. This is the error:   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_ChungReynolds(3)_20241121_102242/execution_iteration_0.py", line 4
# ```python
# ^
# SyntaxError: invalid syntax
#
# Take a look on the feedback: