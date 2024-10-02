 # Name: Custom Metaheuristic
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_mutation',
    {
         'scale': 0.5,
         'elite_rate': 0.1,
         'mutation_rate': 0.25,
         'distribution': 'uniform'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Justification:
# The chosen metaheuristic is named Custom Metaheuristic due to its generic implementation of a genetic mutation operator with specified parameters. This setup allows for the exploration and exploitation phases typical of many evolutionary algorithms, although it does not include crossover as required by some operators like `genetic_crossover`. However, since genetic mutation was mentioned in the task, I've implemented this without crossover to ensure compliance with given instructions.
# The Rastrigin function is selected for benchmarking purposes, which is a common test problem in optimization due to its multimodal nature and challenging search space. This function will be used to evaluate the performance of our metaheuristic across multiple iterations.
# The genetic_mutation operator is chosen because it directly influences the mutation aspect of genetic algorithms, which is essential for maintaining diversity in the population while allowing for local improvements. Parameters scale (0.5) and distribution ('uniform') are selected from parameters_to_take.txt to define the nature of mutations applied during each iteration.
# The probabilistic selector ensures that decisions about applying this mutation operator are made randomly but with a probability that influences how frequently the operator is utilized in the optimization process, which helps balance exploration and exploitation.
# A total of 100 iterations is set for the metaheuristic run to allow sufficient time for convergence towards an optimal solution while avoiding excessive computation times. The verbose mode is enabled to output detailed information about each iteration's progress and final results, aiding in understanding how the algorithm performs under these conditions.