# Name: Hybrid Metaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8163715829892713,
            'self_conf': 2.4205837826174648,
            'swarm_conf': 2.7540164876556066,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7734414284232455,
            'angle': 23.446440125885115,
            'sigma': 0.03794197867421192
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different operators: random sample, swarm dynamic, and spiral dynamic. The `random_sample` operator helps in exploring the search space randomly, while `swarm_dynamic` and `spiral_dynamic` guide the agents towards better solutions more effectively.
# The `probabilistic` selector is used for the `swarm_dynamic` operator to explore the solution space more thoroughly. The `greedy` selector is used for the other two operators to ensure they quickly converge to good solutions.
# This combination allows the algorithm to balance exploration and exploitation, making it suitable for complex optimization problems like the Rastrigin function.

# If you encounter an error, address it as follows:
# File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin_15_20250110_144144/execution_optuna_iteration_2.py", line 1
# ```python
# ^
# SyntaxError: invalid syntax
# ```
# The error indicates that there is a misplaced backtick in your code. Please ensure that the backticks are properly used and removed if unnecessary.