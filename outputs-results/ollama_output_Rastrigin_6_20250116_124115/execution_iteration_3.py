# Name: Hybrid Swarm-Spiral Metaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.3252980981900372,
            'self_conf': 2.2724237627420507,
            'swarm_conf': 2.4566858147012876,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8787844904751487,
            'angle': 17.52041840629585,
            'sigma': 0.10212129667853449
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Swarm-Spiral Metaheuristic combines the strengths of swarm-based and spiral dynamics. 
# The `swarm_dynamic` operator is used to guide the agents towards promising regions, while the `spiral_dynamic` operator helps in exploring new areas efficiently. 
# This hybrid approach aims to balance exploration and exploitation, leading to better convergence properties on the Rastrigin function.

# If you encounter an error, address it as follows:
# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin_6_20250116_124115/execution_iteration_3.py", line 49, in <module>
#     met.run()
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 144, in run
#     self.apply_search_operator(perturbator, selector)
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 100, in apply_search_operator
#     perturbed_solution = operator(solution, **params)
# TypeError: swarm_dynamic() got an unexpected keyword argument 'self_conf'
# The error indicates that the 'swarm_dynamic' function does not accept a 'self_conf' parameter. You need to check the implementation of the 'swarm_dynamic' function and ensure it accepts these parameters correctly.

# Similarly, check the 'spiral_dynamic' function to make sure it accepts the 'radius', 'angle', and 'sigma' parameters. If any of these functions do not accept the required parameters, you will need to modify their implementations accordingly.