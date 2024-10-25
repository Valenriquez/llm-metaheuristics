 # Name: GravitationalSearchAlgorithm
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Algorithm Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is a population-based metaheuristic inspired by the laws of gravity and mass interactions. In this implementation, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. These settings encourage exploration and exploitation in the search space. The selector is set to 'all', meaning all particles will experience gravitational forces affecting their movement towards more optimal regions based on their mass, which could be adjusted by further parameters not specified here.
# The Gravitational Search Algorithm operator is designed to mimic the physical principles of gravity, where objects are attracted to masses with a force proportional to their masses and inversely proportional to the square of the distance between them. In the context of optimization, this translates to attracting solutions towards better regions in the search space based on their fitness values, which are analogous to mass in gravitational interactions. The 'gravity' parameter controls the strength of these attraction forces, while 'alpha' adjusts the influence of the local minima or worse solutions. This approach aims to balance between exploitation (moving towards known good areas) and exploration (discovering new potential areas), which is crucial for escaping local optima and finding global optima in complex multimodal functions like Rastrigin's function used here.