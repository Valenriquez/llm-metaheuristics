from execution_iteration_0 import fitness_array as fitness0
from execution_iteration_1 import fitness_array as fitness1
from execution_iteration_2 import fitness_array as fitness2
from execution_iteration_3 import fitness_array as fitness3
from execution_iteration_4 import fitness_array as fitness4
from execution_iteration_5 import fitness_array as fitness5
from execution_iteration_6 import fitness_array as fitness6
from execution_iteration_7 import fitness_array as fitness7

import numpy as np
from customhys import visualisation as vis

all_fitness = [fitness0,fitness1, fitness2,fitness3, fitness4, fitness5, fitness6, fitness7]

print(all_fitness)
steps = [x for x in range(len(all_fitness))]  

def calculate_performance(fitness_array):
    performances = []
    for iteration_fitness in fitness_array:
        if iteration_fitness.size > 0: 
            med = np.median(iteration_fitness)
            iqr = np.percentile(iteration_fitness, 75) - np.percentile(iteration_fitness, 25)
            performance_metric = med + iqr
            performances.append(performance_metric)
        else:
            performances.append(None) 
    return performances


performances = calculate_performance(all_fitness)
vis.show_performance_overview(steps, all_fitness, performances)
