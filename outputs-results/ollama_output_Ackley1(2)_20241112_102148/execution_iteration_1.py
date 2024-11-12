# Name: ackley_metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) 
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy', 
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'spiral_dynamic', # Changed to spiral dynamic
        {
            'radius': 0.95,  # changed to 0.95 from 0.9,
            'angle': 22.5, 
            'sigma': 0.1
        },
        'metropolis' # changed to metropolis from greedy
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 
# The Ackley function is a multimodal minimization problem that can be used to test the performance of metaheuristic algorithms.
# In this case, we use the Ackley1 function with 2 dimensions.
# The `random_flight` operator uses a Levy flight strategy to search for the optimal solution in the parameter space.
# The `spiral_dynamic` operator uses a spiral pattern to explore the parameter space and converge towards the optimal solution.
# By combining these two operators, we can take advantage of their respective strengths and improve the overall performance of the metaheuristic algorithm.
#
# To get a smaller fitness solution, we need to adjust the parameters of the operators. For example, we can increase the `scale` parameter in the `random_flight` operator
# or decrease the `radius` parameter in the `spiral_dynamic` operator. However, this may require careful tuning and experimentation to achieve optimal results.
#
# One possible approach is to use a combination of heuristics such as adaptive mutation, tabu search, and simulated annealing to guide the search towards the optimal solution.
#
# Additionally, we can incorporate additional features such as parallel processing, dynamic niching, and exploitation-exploitation trade-off to improve the overall performance
# of the metaheuristic algorithm. However, this will require more complex modifications to the algorithm and may increase the computational cost.