 # Name: Adaptive Random Search (ARS)
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1 - Random Flight
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'levy',
                'beta': 1.0
            },
            'probabilistic'
            ),
            (  
            # Search operator 2 - Local Random Walk
            'local_random_walk',
            {
                'probability': 0.8,
                'scale': 0.5,
                'distribution': 'gaussian'
            },
            'probabilistic'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Explanation and Justification:
1. **Name and Purpose**: The metaheuristic is named Adaptive Random Search (ARS), which aims to dynamically adjust its search strategy based on the problem characteristics, leveraging both explorative random flight mechanisms and local intensification via random walks.
2. **Import Statements**: Necessary imports for the benchmark function and metaheuristic functionalities are included. This setup ensures access to predefined functions and classes needed for the experiment.
3. **Benchmark Function Setup**: A Sphere function with dimension 2 is chosen as it is a common testbed, suitable for evaluating search algorithms due to its unimodal nature but with a complex landscape that requires adaptive exploration.
4. **Metaheuristic Initialization**: The metaheuristic framework starts by initializing the benchmark problem and defining the set of operators along with their parameters. Here, two main operators are included:
   - **Random Flight**: This operator uses a levy distribution to encourage long-range jumps, which is beneficial for escaping local minima. Parameters include scale (0.5) and beta (1.0), where beta influences the heaviness of the tail in the levy distribution.
   - **Local Random Walk**: Designed to perform more localized searches by mutating solutions with a gaussian distribution, this operator helps in refining the search around the current best solution. Parameters are probability (0.8) and scale (0.5).
5. **Selector Types**: Both operators use 'probabilistic' selection criteria where the decision to apply an operator is based on predefined probabilities, making them adaptive and less rigid than deterministic selectors.
6. **Run Configuration**: The metaheuristic runs for 200 iterations, allowing both explorative and local search capabilities to operate iteratively. Verbose output is enabled to track the progress and performance of the algorithm in real-time.
7. **Output**: After execution, the best solution found by the metaheuristic is printed along with its fitness value, providing a clear outcome from the optimization process.
8. **Adaptive Strategy**: The choice of operators and their parameters is justified based on literature suggesting that adaptive random search strategies combined with diverse mutation operators can effectively navigate complex non-linear landscapes.
9. **Code Integrity**: The code strictly adheres to using only the specified operators and parameters from `parameters_to_take.txt`, ensuring methodological consistency and reproducibility of results.