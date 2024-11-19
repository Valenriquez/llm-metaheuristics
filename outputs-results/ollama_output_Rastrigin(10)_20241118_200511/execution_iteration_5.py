### Metaheuristic Algorithm for Optimization

The proposed metaheuristic algorithm combines multiple search operators to address complex optimization problems effectively. This combination allows the algorithm to balance exploration and exploitation strategies, leading to more efficient convergence towards global optima.

#### Components of the Proposed Metaheuristic

1. **Gravitational Search Algorithm (GSA)**
   - **Objective:** Find global optima.
   - **Mechanism:** Uses the principles of gravity and masses to simulate the attraction of particles towards better solutions.
   - **Parameters:**
     - `gravity`: A constant value representing the strength of the gravitational force.
     - `alpha`: An adaptive parameter affecting the mass distribution among particles.

2. **Random Flight (RF)**
   - **Objective:** Explore new regions of the solution space.
   - **Mechanism:** Utilizes a probabilistic approach to generate random movements that help in escaping local minima and exploring distant areas.
   - **Parameters:**
     - `scale`: Determines the extent of movement.
     - `distribution`: Specifies the probability distribution for generating random positions (e.g., Lévy flights).
     - `beta`: Parameter for the Lévy flight distribution.

3. **Local Random Walk (LRW)**
   - **Objective:** Fine-tune around the current best solutions.
   - **Mechanism:** Provides fine-grained exploration in the vicinity of the current solution, ensuring convergence to high-quality solutions.
   - **Parameters:**
     - `probability`: Probability of performing a random walk move.
     - `scale`: Determines the step size for the local moves.
     - `distribution`: Specifies the distribution for generating steps (e.g., uniform).

#### Combination Strategy

- **Exploration:** 
  - The Random Flight operator enhances exploration by allowing particles to venture into new areas where global optima might be hiding.

- **Exploitation:**
  - The Gravitational Search Algorithm and Local Random Walk help in exploiting the promising regions identified during exploration.
  - By leveraging different stages of the optimization process, each operator ensures a balanced mix of thorough exploration and efficient exploitation.

#### Implementation

The implementation of this metaheuristic involves initializing the population of particles, applying the combined search operators iteratively for a specified number of iterations (100 in this case), and monitoring the progress. The algorithm adapts parameters dynamically based on the performance during each iteration.

#### Results

The combination of these operators effectively balances exploration and exploitation, leading to improved convergence rates and better solutions compared to using individual operators separately.

### Example Implementation

```python
import numpy as np
from metaheuristic import Metaheuristic, gravitational_search, random_flight, local_random_walk

# Define the problem (Rastrigin function)
def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

# Problem dimensions
dim = 10

# Initialize Metaheuristic with multiple operators
operators = [
    (gravitational_search, {'gravity': 1.0, 'alpha': 0.02}, 'all'),
    (random_flight, {'scale': 1.0, 'distribution': 'levy', 'beta': 1.5}, 'all'),
    (local_random_walk, {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'all')
]

# Create Metaheuristic instance
metaheuristic = Metaheuristic(dim, rastrigin, operators, num_iterations=100)

# Run the algorithm
best_solution, best_fitness = metaheuristic.run()

print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
```

This implementation showcases how to integrate and balance multiple search operators in a single metaheuristic framework to solve optimization problems more effectively.