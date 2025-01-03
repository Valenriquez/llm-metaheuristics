# Name: HybridMetaHeuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # This is the selected problem, the problem may vary depending on the case.
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7989575743359129,
            'angle': 18.647765704096585,
            'sigma': 0.021759009801701302
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5401274414626851,
            'self_conf': 2.454110566851833,
            'swarm_conf': 2.3605588754783096,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines the simplicity of random sampling with the efficiency of spiral dynamic and swarm dynamics. The use of metropolis selector in spiral dynamic helps escape local minima, while probabilistic selector in swarm dynamic allows for diverse exploration. This combination aims to balance between exploitation and exploration to effectively solve the Rastrigin function problem.

# If you encounter an error, address it as follows:
# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin(3)_20241230_171621/execution_iteration_0.py", line 50, in <module>
#     print('x_best = {}, f_best = {}'.format(*met.get_solution()))
#                                              ^^^^^^^^^^^^^^^^^^
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 173, in get_solution
#     return self.historical['position'][-1], self.historical['fitness'][-1]
#            ~~~~~~~~~~~~~~~^^^^^^^^^^^^
# KeyError: 'position'
# .
# Explanation:
# The error indicates that the historical dictionary does not contain a key for 'position'. This could be due to an issue in the metaheuristic implementation where the position is not being recorded or stored correctly. Ensure that the metaheuristic class captures and stores the best position found during the optimization process.