# Name: Hybrid Swarm-Gravitational Search Algorithm (HSGSA)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Gravitational Search Algorithm (GSA)
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    (
        # Search operator 2: Swarm Dynamic (PSO-like)
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 1. The Hybrid Swarm-Gravitational Search Algorithm (HSGSA) combines the strengths of both Gravitational Search Algorithm (GSA) and Particle Swarm Optimization (PSO).
# 2. GSA is effective in escaping local minima but may converge slowly. PSO, on the other hand, is known for its fast convergence but can get stuck in local optima.
# 3. By integrating both approaches, HSGSA aims to balance exploration and exploitation, leading to improved search performance.
# 4. The use of 'all' as the selector means that all particles and gravitational objects are considered in each iteration, ensuring comprehensive coverage of the search space.