# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the hybrid operator with multiple strategies
heur = [
    (  # Search operator 1: Random Search with Gaussian distribution and Metropolis selector
        'random_search',
        {
            'scale': 0.5,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        # Search operator 2: Local Random Walk with Uniform distribution and Probabilistic selector
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Swarm Dynamic with Gaussian distribution and All selector
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'all'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
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
# The hybrid metaheuristic combines three different search operators to explore the solution space more effectively.
# It uses Random Search with Gaussian distribution to escape local optima, Local Random Walk to refine solutions,
# and Swarm Dynamic with Gaussian distribution for collective exploration. Each operator is equipped with a specific
# selector (Metropolis, Probabilistic, All) to control how new solutions are accepted or rejected.

# Addressing the error:
# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin(3)_20241229_214524/execution_iteration_0.py", line 58, in <module>
#     fitness.append(met.historical['fitness'])
#     ^^^^^^^
# NameError: name 'fitness' is not defined

# The error was due to the variable 'fitness' being used without being defined. It has been corrected by defining it before using it in the loop.