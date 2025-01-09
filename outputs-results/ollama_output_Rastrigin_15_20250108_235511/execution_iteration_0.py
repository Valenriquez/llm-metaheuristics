# Name: Hybrid Swarm Optimization
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
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
#met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the Swarm Dynamic operator with the Spiral Dynamic operator to explore the search space more effectively. The Swarm Dynamic operator helps in aggregating solutions, while the Spiral Dynamic operator provides a diverse exploration strategy. The use of probabilistic selectors ensures that both operators are used adaptively during the search process.

# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin_15_20250108_235511/execution_iteration_0.py", line 58, in <module>
#     met.run()
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 144, in run
#     self.apply_search_operator(perturbator, selector)
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 100, in apply_search_operator
#     exec('Operators.' + operator_name + '(self.pop,' + operator_params)
#   File "<string>", line 1, in <module>
# TypeError: central_force_dynamic() got an unexpected keyword argument 'factor'
# The error indicates that the 'central_force_dynamic' operator was called with a parameter 'factor' that it does not accept. This suggests that there might be a mismatch between the parameters expected by the operator and those provided in the metaheuristic setup.