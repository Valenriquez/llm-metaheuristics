 # Name: GravitationalSearchAlgorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin2
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Algorithm Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the principles of gravitational force and mass interactions. In this implementation, we use the GSA with specific parameters for gravity (1.0) and alpha (0.02). We select 'metropolis' as the selector because it aligns with probabilistic search strategies that can guide the algorithm towards better solutions. The Rastrigin function is chosen due to its multimodal nature, which allows multiple local minima, making it a suitable test for metaheuristic algorithms capable of exploring diverse solution spaces.
# The GSA operator uses gravity and alpha parameters to simulate the gravitational force between masses, promoting exploration or exploitation based on these settings. The selector 'metropolis' ensures that the algorithm can switch between different search strategies dynamically, enhancing its ability to escape local minima and converge towards a global optimum. This setup is intended to balance between convergence speed and accuracy in solving complex optimization problems like the Rastrigin function with multiple peaks.