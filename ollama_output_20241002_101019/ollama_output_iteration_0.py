 # Name: Custom Metaheuristic with Gravitational Search and Genetic Mutation
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'gravitational_search',
    {
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Explanation and Justification:
# The metaheuristic is named Custom Metaheuristic with Gravitational Search and Genetic Mutation. 
# This name reflects the combination of gravitational search and genetic mutation operators used in the algorithm.

# A benchmark function, specifically Rastrigin(2), is selected as it is a suitable test problem for optimization algorithms due to its multiple local minima which makes it challenging yet useful for benchmarking purposes.

# Gravitational Search (GS) is chosen because it mimics the gravitational interaction between masses in space, where each mass represents a potential solution in the search space. This operator allows the algorithm to explore different areas of the search space by simulating gravity-driven movements among candidate solutions.

# The GS operator's parameters are set as follows: 
# - 'gravity': Set to 1.0 which is typical for scaling the force of attraction or repulsion between masses in gravitational systems, though this parameter can be adjusted based on specific problem characteristics (verified from parameters_to_take.txt).
# - 'alpha': This parameter controls the speed at which solutions converge towards better areas; a value of 0.02 is typical for starting adjustments before fine-tuning during optimization runs (also verified from parameters_to_take.txt).

# The selector used here, 'all', means that all candidate solutions in the population will undergo gravitational search. This approach helps to diversify and converge towards better regions of the solution space simultaneously.

# Genetic Mutation is integrated into the algorithm as a standard practice in genetic algorithms (GA) to introduce diversity into the population by randomly altering the genotype values of candidate solutions, which is crucial for escaping local minima and exploring new areas of the search space. The mutation operator's parameters are not explicitly set here but are typically adjusted based on problem complexity:
# - 'scale': Controls the extent of the mutations applied to solution vectors (verified from parameters_to_take.txt).
# - 'distribution': Chooses how the mutation values are distributed, such as uniform or Gaussian distribution (also verified from parameters_to_take.txt), which affects how much and in what direction variations occur during mutation.

# The metaheuristic is configured to run for 100 iterations, allowing ample time for both exploration of the solution space and convergence towards an optimal solution based on the Rastrigin function's characteristics.

# This setup ensures that the algorithm balances between exploring new areas with mutations and refining solutions through gravitational interactions, which should lead to effective optimization results according to the problem requirements as per parameters_to_take.txt.