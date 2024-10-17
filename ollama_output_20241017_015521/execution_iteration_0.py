 # Name: Adaptive Random Search (ARS) Metaheuristic
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Random Flight
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'levy',
                'beta': 1.2
            },
            'metropolis'
            ),
            (  
            # Search operator 2: Local Random Walk
            'local_random_walk',
            {
                'probability': 0.6,
                'scale': 0.8,
                'distribution': 'gaussian'
            },
            'probabilistic'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Short explanation and justification:
The Adaptive Random Search (ARS) metaheuristic is designed to explore the search space using two primary operators: Random Flight and Local Random Walk. These operators are chosen based on their ability to balance exploration and exploitation, which is crucial for optimizing complex multi-modal functions like the Rastrigin function in this case.

1. **Random Flight**: This operator uses a Levy distribution to allow for long jumps that can help escape local minima. The scale parameter is set to 0.5, and beta to 1.2 to maintain an appropriate balance between exploration and exploitation.

2. **Local Random Walk**: This operator employs Gaussian distribution for small step sizes, ensuring more focused movement around the current best solution. The probability is set to 0.6, and scale to 0.8 to allow for both local search and occasional random jumps.

Both operators are selected with 'metropolis' as their selector, which aligns with the probabilistic nature of these movements in the optimization process. The metaheuristic runs for 150 iterations, providing ample opportunity for the population to adapt and converge on an optimal solution. This approach ensures a comprehensive exploration of the search space while maintaining the ability to fine-tune solutions through local searches.