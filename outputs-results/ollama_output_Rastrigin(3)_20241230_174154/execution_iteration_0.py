```python
# Name: Hybrid Metaheuristic for Global Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.41488118297369725,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        # Search operator 2: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.34553941327815024,
            'alpha': 1.3602157228295886,
            'beta': 1.8975963889372307,
            'dt': 2.0518781300027085
        },
        'all'
    ),
    (
        # Search operator 3: Differential Mutation
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 0.6132345908399812
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# The hybrid metaheuristic combines three different search operators to leverage their strengths for global optimization.
# 1. Random Search helps in exploring the solution space and escaping local optima.
# 2. Central Force Dynamic mimics the behavior of charged particles in a physical system, which can efficiently explore complex landscapes.
# 3. Differential Mutation is effective for fine-tuning solutions and handling multimodal problems.
# This combination aims to balance exploration and exploitation, leading to improved performance on the Rastrigin function.
```