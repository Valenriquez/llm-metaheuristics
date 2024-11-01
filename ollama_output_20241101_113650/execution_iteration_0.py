 # Name: Custom Metaheuristic with Levy Distribution
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'levy'
        },
        'metropolis'
    ),
    (  
        'random_sample',
        {
            'selector': 'probabilistic'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# This metaheuristic uses a custom approach with two operators to explore the Rastrigin function. 
# The first operator, 'swarm_dynamic', is configured with specific parameters for its behavior. 
# It utilizes an inertial version of swarm dynamics, where particles adjust their movement based on local information and a levy distribution, which is known for its ability to jump across large distances effectively in search spaces.
# The selector for this operator is set to 'metropolis', which aligns with the probabilistic nature of the levy distribution, enhancing global exploration while allowing occasional jumps that might lead to better solutions.
# The second operator, 'random_sample', uses a simple probabilistic selector called 'probabilistic'. This allows it to randomly sample solutions from the population, increasing diversity and potentially leading to improved convergence by avoiding local minima.
# Both operators are designed with parameters justified from the benchmark requirements, focusing on balancing exploration (levy distribution) and exploitation (metropolis selection).