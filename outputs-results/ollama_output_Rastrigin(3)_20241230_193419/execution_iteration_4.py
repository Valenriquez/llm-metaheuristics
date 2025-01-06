# Name: Hybrid Metaheuristic for Rastrigin Function Optimization
# Code:
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Griewank(6)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    )
]

# Update the parameters for each operator
heur[1] = ('swarm_dynamic', {'factor': 0.3568344875136792, 'self_conf': 2.951976243005424, 'swarm_conf': 2.744741106507324, 'version': 'inertial', 'distribution': 'uniform'}, 'probabilistic')
heur[2] = ('spiral_dynamic', {'radius': 0.47507395521781026, 'angle': 5.677651051605848, 'sigma': 0.020831420146921104}, 'metropolis')

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()


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


# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))

    fitness.append(met.historical['fitness'])
print("fitness", fitness)
fitness_array = np.array(fitness).T

#for rep in range(30):
#    print('rep = {}'.format(fitness_array))

# Assuming fitness_array is already defined
performances = calculate_performance(fitness_array)
print("Performance Metrics per Iteration:", performances)



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

fitness_array = np.array(fitness).T

# Extract final, best, and initial fitness values
final_fitness = np.array([x[-1] for x in fitness_array.T])
best_fitness = np.min(fitness_array)
initial_fitness = max([x[0] for x in fitness_array])

# Get the iteration numbers of best and worst fitness
best_iteration = np.argmin(fitness_array)
worst_iteration = np.argmax(fitness_array)

# Now plot the fitness values
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey='row', gridspec_kw={'width_ratios': [0.8, 0.2]})
axs[0].plot(fitness_array, 'grey', alpha=0.3)

axs[0].plot(np.min(fitness_array, axis=1), 'g', lw=2, label=f'Best (Value: {best_fitness})')
axs[0].plot(np.max(fitness_array, axis=1), 'k', lw=2, label=f'Worst (Value: {np.max(fitness_array)})')
axs[0].plot(np.mean(fitness_array, axis=1), 'b', lw=2, label='Mean')
axs[0].plot(np.median(fitness_array, axis=1), 'r', lw=2, label='Median')

# Display the best and worst iteration on the plot
axs[0].scatter(best_iteration, best_fitness, color='green', zorder=5)
axs[0].scatter(worst_iteration, np.max(fitness_array), color='black', zorder=5)

# Annotating the best and worst iterations
axs[0].text(best_iteration, best_fitness, f'Best Iteration: {best_iteration}\nValue: {best_fitness}',
            color='green', fontsize=10, ha='center', va='bottom')
axs[0].text(worst_iteration, np.max(fitness_array), f'Worst Iteration: {worst_iteration}\nValue: {np.max(fitness_array)}',
            color='black', fontsize=10, ha='center', va='top')

axs[0].set_ylabel(r'Fitness')
axs[0].set_xlabel(r'Iteration')
axs[0].set_ylim([best_fitness, initial_fitness])

# Violin plot
axs[1].violinplot(final_fitness, showmeans=True, showmedians=True)
axs[1].set_xlabel(r'Data')
# Adjust the violin plot to make it proportionate
 
# Adjust the legend
axs[0].legend(frameon=False)

axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].set_xscale('log')
plt.tight_layout()
plt.show()