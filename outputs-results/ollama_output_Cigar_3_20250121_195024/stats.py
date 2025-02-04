from execution_iteration_0 import final_fitness as fitness0
from execution_iteration_1 import final_fitness as fitness1
from execution_iteration_2 import final_fitness as fitness2
from execution_iteration_3 import final_fitness as fitness3
from execution_iteration_4 import final_fitness as fitness4
from execution_iteration_5 import final_fitness as fitness5
from execution_iteration_6 import final_fitness as fitness6
from execution_iteration_7 import final_fitness as fitness7
from execution_iteration_8 import final_fitness as fitness8

import numpy as np
from customhys import visualisation as vis

# Combine all fitness arrays into
all_fitness = [fitness0, fitness1, fitness2, fitness3]

steps = [x for x in range(len(all_fitness))]  # Generates [0, 1, 2, ..., 9]

for i, fitness in enumerate(all_fitness):
    print(f"Shape of fitness{i}: {fitness.shape}")

def calculate_performance(fitness_array):
    performances = []
    # Loop through each iteration (row in fitness_array)
    for iteration_fitness in fitness_array:
        if iteration_fitness.size > 0:  # Ensure there are fitness values for this iteration
            med = np.median(iteration_fitness)
            iqr = np.percentile(iteration_fitness, 75) - np.percentile(iteration_fitness, 25)
            performance_metric = med + iqr
            performances.append(performance_metric)
        else:
            performances.append(None)  # Placeholder for missing data
    return performances

performances = calculate_performance(all_fitness)
vis.show_performance_overview(steps, all_fitness, performances)