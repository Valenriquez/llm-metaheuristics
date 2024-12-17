# Name: Hybrid Randomized Search with Genetic Algorithm
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Adjust the number of agents based on the dimension size
num_agents = 2 if prob['dim'] <= 5 else int(prob['dim']) ** 2

heur = [
    (  # Search operator 1: Genetic Algorithm - Crossover and Mutation
        'crossover',
        {
            'num_parents_mating': int(num_agents / 4), # Number of parents to mate for each generation
        },
        'greedy'
    ),
    (
        'mutation',
        {
            'random_mutation_probability': 0.1, # Probability of mutating a gene randomly
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines Genetic Algorithm (GA) operators such as crossover and mutation with a greedy selector. 
# The GA is initialized with a certain number of agents based on the problem dimension size.
# During each iteration, the GA performs mating to create offspring, followed by random mutations for genetic diversity.
# The search operator applies these changes greedily to explore potential solutions in the solution space.
# The metaheuristic runs multiple iterations and records the best fitness found in each run. 
# By running it 30 times, we can get an average performance and assess the robustness of the approach.

# If an error occurs, please check if the problem dictionary contains a key 'dim' representing the dimension size.
# Adjust the error handling accordingly to handle missing keys gracefully.