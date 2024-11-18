```python
# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.rastrigin(10)  # WRITE IT EXACTLY AS GIVEN, BE CAREFUL
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```

# Short explanation and justification:
# - The `HybridMetaheuristic` combines multiple search operators (`random_sample`, `gravitational_search`, and `local_random_walk`) to leverage their strengths.
# - `random_sample` provides a broad exploration of the search space.
# - `gravitational_search` utilizes the concept of gravitational attraction, which helps in finding promising regions quickly.
# - `local_random_walk` ensures that we do not get stuck in local minima by exploring the neighborhood of the current solution.
# - The combination aims to balance global and local search strategies for better performance on the Rastrigin benchmark function.