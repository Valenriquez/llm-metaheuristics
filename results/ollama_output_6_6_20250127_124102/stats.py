from execution_iteration_0 import fitness_array as fitness0
from execution_iteration_1 import fitness_array as fitness1
from execution_iteration_2 import fitness_array as fitness2
from execution_iteration_3 import fitness_array as fitness3
from execution_iteration_4 import fitness_array as fitness4
from execution_iteration_5 import fitness_array as fitness5
from execution_iteration_6 import fitness_array as fitness6
from execution_iteration_7 import fitness_array as fitness7
from execution_iteration_8 import fitness_array as fitness8
from execution_iteration_9 import fitness_array as fitness9
from execution_iteration_10 import fitness_array as fitness10
from collections import Counter

import numpy as np
from customhys import visualisation as vis

all_fitness = [fitness0,fitness1, fitness2, fitness3, fitness4, fitness5, fitness6, fitness7, fitness8, fitness9, fitness10]

print(all_fitness)
steps = [x for x in range(len(all_fitness))]  # Generates [0, 1, 2, ..., 9]

def calculate_performance(fitness_array):
    performances = []
    # Loop through each iteration (row in fitness_array)
    for iteration_fitness in fitness_array:
        if iteration_fitness.size > 0:  # Ensure there are fitness values for this iteration
            med = np.median(iteration_fitness)
            iqr = np.percentile(iteration_fitness, 75) - np.percentile(iteration_fitness, 25)
            std_dev = np.std(iteration_fitness)
            print("std_dev", std_dev)
            performance_metric = med + iqr
            print("performance_metric", performance_metric)
            performances.append(performance_metric)
        else:
            performances.append(None)  # Placeholder for missing data
    return performances

# Flatten all_fitness arrays into a single list for analysis
flattened_fitness = np.concatenate(all_fitness)

# Find the lowest value
lowest_value = np.min(flattened_fitness)

# Count occurrences of each value
value_counts = Counter(flattened_fitness)

# Find the lowest value that repeats the most
most_repeated_lowest_value = min(
    value_counts, key=lambda x: (-value_counts[x], x)  # Sort by frequency desc, then value asc
)

# Print results
print(f"Lowest value: {lowest_value}")
print(f"Lowest value that repeats the most: {most_repeated_lowest_value}")



performances = calculate_performance(all_fitness)
vis.show_performance_overview(steps, all_fitness, performances)


