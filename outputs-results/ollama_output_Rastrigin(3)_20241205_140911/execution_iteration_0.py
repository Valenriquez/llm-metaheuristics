When creating a metaheuristic take into account: You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response, all outputs must be plain text. Use only the benchmark_function and its' dimension provided. Remember that you should add more agents depending on the size of the dimension, the default agents are 2, however if there is a dimension as 5, you should add 2 more agents, and so on and so forth. If there is a 10 dimension you should add 100 agents, and 15 dimension too.

# Name: Hybrid Swarm Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Increase number of agents based on dimension
num_agents = 2 if prob['dimension'] == 1 else prob['dimension']

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {
            'parameter1': value1,
            'parameter2': value2,
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial",
            'distribution': "uniform"
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Hybrid Swarm Optimization combines random sampling with swarm dynamics to explore the search space more effectively. This approach leverages the exploration power of random sampling while benefiting from the exploitation capabilities of swarm intelligence.
```
obj['response']
```