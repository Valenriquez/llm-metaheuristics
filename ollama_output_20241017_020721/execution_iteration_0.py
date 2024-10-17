 # Name: GravitationalSearchOptimization
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'levy',
                'beta': 1.5
            },
            'probabilistic'
            ),
            (  
            'gravitational_search',
            {
                'gravity': 0.8,
                'alpha': 0.02
            },
            'metropolis'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Short explanation and justification:
# The metaheuristic is named GravitationalSearchOptimization (GSO). GSO combines the exploration capabilities of random flight operators with the exploitation power of gravitational search operators to efficiently explore the solution space for continuous optimization problems like the Rastrigin function. The chosen parameters are set as follows:
# 1. 'random_flight' operator is used with a scale factor of 0.5, distribution set to 'levy', and beta value of 1.5. This operator introduces random perturbations into the search space, aiding in global exploration.
# 2. 'gravitational_search' operator uses a gravity constant of 0.8 and an alpha coefficient of 0.02. This operator simulates gravitational forces between particles to guide the search towards better solutions, enhancing local exploitation.
# The probabilistic selector is chosen for both operators to ensure that their actions are selected based on predefined probabilities, allowing for more controlled exploration-exploitation trade-offs. The total number of iterations is set to 150 to balance convergence speed and exploration breadth.