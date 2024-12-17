# Name: Adaptive Dynamic Metaheuristic (ADM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{self.benchmark_function}({self.dimensions})
prob = fun.get_formatted_problem()

def select_operators(num_agents):
    if num_agents <= 5:
        return ["random_search", "central_force_dynamic", "firefly_dynamic"]
    elif num_agents <= 10:
        return ["random_search", "central_force_dynamic", "random_flight", "differential_mutation"]
    else:
        return ["genetic_crossover", "genetic_mutation", "spiral_dynamic", "swarm_dynamic"]

def apply_operators(operators_list, population):
    new_population = []
    for agent in population:
        operator = random.choice(operators_list)
        new_agent = apply_operator(agent, operator)
        new_population.append(new_agent)
    return new_population

num_agents = {self.dimensions} + 2
selected_operators = select_operators(num_agents)

heur = [
    (  # Search operator 1
        '[operator_name]',
        {
            'parameter1': value1,
            'parameter2': value2,
            more parameters as needed
        },
        '[selector_name]'
    ),
    (
        '[operator_name]',
        {
            'parameter1': value1,
            'parameter2': value2,
            ... more parameters as needed
        },
        '[selector_name]'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# This metaheuristic uses a dynamic approach to select operators based on the number of agents.
# The number of agents increases with the dimension size to ensure adequate exploration of the search space.
# The selection of operators is adaptive, adjusting based on the problem's complexity.
# This ensures efficient use of computational resources and better performance for larger problems.