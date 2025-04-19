import numpy as np
from customhys import visualisation as vis
from collections import Counter

"""
Evaluates and visualizes performance based on fitness values.

Parameters:
- fitness_values: list of float
- vis: visualization object that has method show_performance_overview(steps, fitness, performance)
- calculate_performance: function to compute overall performance from fitness list
- lowest_fitness: function to find lowest fitness values

Returns:
- performances: computed performance values
- lowest: lowest fitness found
"""

def calculate_performance(fitness_array):
    performances = []
    for iteration_fitness in fitness_array:
      med = np.median(iteration_fitness)
      iqr = np.percentile(iteration_fitness, 75) - np.percentile(iteration_fitness, 25)
      std_dev = np.std(iteration_fitness)
      print("std_dev", std_dev)
      performance_metric = med + iqr
      print("performance_metric", performance_metric)
      performances.append(performance_metric)
    return performances

def lowest_fitness(fitness_array):
  flattened_fitness = np.concatenate(fitness_array)
  lowest_value = np.min(flattened_fitness)
  value_counts = Counter(flattened_fitness)

  most_repeated_lowest_value = min(
      value_counts, key=lambda x: (-value_counts[x], x)
  )
  return most_repeated_lowest_value

def visualization(all_fitness):
    #all_fitness = [fitness0,fitness1, fitness2,fitness3, fitness4, fitness5, fitness6, fitness7, fitness8, fitness9, fitness10, fitness11, fitness12]
    steps = [x for x in range(len(all_fitness))]

    performances = calculate_performance(all_fitness)
    print(performances)
    vis.show_performance_overview(steps, all_fitness, performances)

    print(f"Lowest values: {lowest_fitness(all_fitness)}")