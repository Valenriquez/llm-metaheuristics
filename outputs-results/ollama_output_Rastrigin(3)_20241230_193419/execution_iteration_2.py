# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7921493637383673,
            'self_conf': 1.3785178317108597,
            'swarm_conf': 1.734394983111914,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6671145344570681,
            'angle': 19.309820701269466,
            'sigma': 0.2804217612219863
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []

# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
print("fitness_array", fitness_array)
final_fitness = np.array([x[-1] for x in fitness_array.T])
best_fitness = np.min(fitness_array)
initial_fitness = max([x[0] for x in fitness_array])

# Calculate key metrics
mean_final_fitness = np.mean(final_fitness)
std_final_fitness = np.std(final_fitness)
best_final_fitness = np.min(final_fitness)
worst_final_fitness = np.max(final_fitness)
mean_initial_fitness = np.mean([x[0] for x in fitness_array])
best_initial_fitness = np.min([x[0] for x in fitness_array])

# Save metrics for reporting or comparison

print("mean_final_fitness", mean_final_fitness)
print("std_final_fitness", std_final_fitness)
print("best_final_fitness", best_final_fitness)
print("worst_final_fitness", worst_final_fitness)
print("mean_initial_fitness", mean_initial_fitness)
print("best_initial_fitness", best_initial_fitness)

results = {
    "mean_final_fitness": mean_final_fitness,
    "std_final_fitness": std_final_fitness,
    "best_final_fitness": best_final_fitness,
    "worst_final_fitness": worst_final_fitness,
    "mean_initial_fitness": mean_initial_fitness,
    "best_initial_fitness": best_initial_fitness,
}  
# Short explanation and justification:
# This hybrid metaheuristic combines the Swarm Dynamic and Spiral Dynamic operators.
# The Swarm Dynamic operator is selected with a probabilistic selector to allow for exploration of the solution space using the specified parameters.
# The Spiral Dynamic operator is chosen with a greedy selector, aiming to exploit local optima efficiently using the given parameters.
# This combination aims to balance both exploration and exploitation, potentially leading to better solutions for complex optimization problems.