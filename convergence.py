import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example data
iterations = np.arange(1, 61)
runs = np.array([
    # Run 1 fitness values
    [0.9, 0.8, 0.7, ..., 0.1],
    # Run 2 fitness values
    [0.95, 0.85, 0.75, ..., 0.15],
    # ...
    # Run 50 fitness values
    [0.92, 0.82, 0.72, ..., 0.12]
])

# Calculate statistics
best = np.min(runs, axis=0)
worst = np.max(runs, axis=0)
mean = np.mean(runs, axis=0)
median = np.median(runs, axis=0)

# Plot
plt.figure(figsize=(12, 5))

# Convergence plot
plt.subplot(1, 2, 1)
for run in runs:
    plt.plot(iterations, run, color='gray', alpha=0.3)
plt.plot(iterations, best, color='green', label='Best')
plt.plot(iterations, worst, color='black', label='Worst')
plt.plot(iterations, mean, color='blue', label='Mean')
plt.plot(iterations, median, color='red', label='Median')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()

# Violin plot
plt.subplot(1, 2, 2)
sns.violinplot(data=runs[:, -1], orient='h')
plt.xlabel('Data')

plt.tight_layout()
plt.show()