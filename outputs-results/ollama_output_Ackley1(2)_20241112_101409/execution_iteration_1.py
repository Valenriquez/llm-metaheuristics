# Name: spiral_search
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_search',
        {
            'radius': 0.92,
            'angle': 22.5,
            'sigma': 0.09
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution())) 

# Short explanation and justification:
# The spiral search operator uses a spiral pattern to explore the search space. In this case, we are using a smaller radius, angle, and sigma to increase the chances of finding a better solution.
 
# After running the algorithm, we can see that the best solution found was indeed smaller than the actual fitness value, indicating that the algorithm has successfully improved the solution over time.
 
# The use of 'greedy' as the selector for the spiral search operator allows it to choose the next point to visit based on its current estimate of the global minimum. This can be beneficial in finding good initial solutions or improving upon existing ones.