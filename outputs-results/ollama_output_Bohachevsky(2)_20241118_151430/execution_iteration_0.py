# Name: Harmony Search with Gravitational Search Algorithm (HS-GSA)

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(3)  # This is the selected problem, the problem may vary depending on the case.
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
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# Harmony Search (HS) is an optimization algorithm inspired by the improvisation process of musicians in a music ensemble. It mimics the behavior of musicians adjusting their pitches harmoniously to achieve better musical outcomes.
# Gravitational Search Algorithm (GSA) is another optimization technique that models the law of gravity and Newton's laws of motion for search space exploration, exploiting the idea of mass distribution and gravitational pull for seeking the optimal solution.
# By combining HS with GSA, we aim to leverage the harmony-building process in HS with the robust global search capability of GSA. The `gravitational_search` operator helps guide the solution particles towards a better region by simulating gravity, while the `random_sample` operator introduces diversity and randomness to escape local minima. This hybrid approach is expected to enhance the exploration-exploitation trade-off, leading to improved convergence and solution quality.
# The use of the 'greedy' selector ensures that each operator's contribution is evaluated based on its immediate impact, favoring operators that provide better solutions or progress more significantly towards the optimal value.
# The `num_iterations` parameter controls the number of iterations for the metaheuristic to run, providing enough time for the combined search strategy to converge.