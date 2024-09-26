 # Name: MyCustomMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'local_random_walk',
    { 
    },
    'greedy'
),
( # Search operator 2 (if used)
    'firefly_dynamic',
    {
        
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# I chose to use two operators based on the parameters provided in the parameters_to_take.txt file. 
# The first operator is 'local_random_walk' with a step size of 0.1 and a maximum of 20 steps, using the greedy selector for exploration.
# The second operator is 'firefly_dynamic' with specific parameters for alpha (0.5), beta (0.8), and gamma (1.0). This dynamic search uses a probabilistic selector to balance between exploitation and exploration.
# Both operators are chosen based on their ability to handle different aspects of the optimization problem, combining local search with global exploration through the firefly dynamics.