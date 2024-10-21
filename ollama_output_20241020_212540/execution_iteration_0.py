 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    ( # Random Flight Operator
        'random_flight',
        { 
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the Gravitational Search algorithm with the Random Flight mechanism. 
# The Gravitational Search, characterized by its gravity parameter (1.0) and alpha value (0.02), is used to simulate the gravitational force acting on particles within a search space. 
# The purpose of this operator is to explore the solution space by allowing particles to move towards regions with higher potential energy, which corresponds to better solutions in optimization problems.
# On the other hand, the Random Flight mechanism, defined by its scale (1.0), distribution type ('levy'), and beta value (1.5), introduces a stochastic element into the search process. 
# The 'levy' distribution used here biases the movement towards more diverse parts of the search space, which is beneficial for escaping local minima and exploring new areas in the search for better solutions.
# The selector 'probabilistic' ensures that both operators will operate with a probabilistic approach, allowing some elements to follow deterministic rules while others take random steps based on their probability settings. 
# This combination of gravitational attraction towards good regions and random exploratory movements helps in balancing between exploration and exploitation, which is crucial for optimizing complex functions such as the Rastrigin function used here.