# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.ackley(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'all'
        
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The HybridMetaheuristic combines Gravitational Search (GSA) and Random Flight (RF) to create a hybrid approach for solving optimization problems.
# GSA is chosen due to its capability in exploring the search space effectively, while RF adds local exploration through random walks which can help in fine-tuning the solution.
# The 'greedy' selector is used for GSA as it tends to move towards solutions that offer immediate improvements, enhancing global exploration.
# The 'all' selector is employed for RF to consider all potential moves during each iteration, increasing the likelihood of finding a better local optimum.
# This combination aims to balance between exploitation and exploration, potentially leading to improved performance on a variety of benchmark functions.