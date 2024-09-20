import random
import math
import numpy as np

class OperatorsError(Exception):
    pass


#Random Search (RS) does not need any presentation because it is the seed of most of the existing metaheuristics.
#
def random_search(iterations, min_x, max_x):
    """
    Perform random search to find the minimum of the objective function.
    
    :param iterations: Number of random points to try
    :param min_x: Minimum value for x
    :param max_x: Maximum value for x
    :return: Best x found and its corresponding y value
    """
    best_x = None
    best_y = float('inf')  # Initialize with positive infinity

    for _ in range(iterations):
        # Generate a random x value within the specified range
        x = random.uniform(min_x, max_x)
        
        # Evaluate the objective function at this point
        y = objective_function(x)
        
        # If this y is better (lower) than our current best, update best_x and best_y
        if y < best_y:
            best_x = x
            best_y = y
    
    return best_x, best_y
 

# Set the parameters for both searches
iterations = 1000
min_x, max_x = -10, 10

# Run the random search
random_best_x, random_best_y = random_search(iterations, min_x, max_x)
 
# Print the results
print(f"Random Search - Best x: {random_best_x:.6f}, Best y: {random_best_y:.6f}")
print(f"Actual minimum - x: 0, y: 5")


def _random_levy(size, beta=1.5):
    """
    This is an internal method to draw a random number (or array) using the Levy stable distribution via the
    Mantegna's algorithm.
        R. N. Mantegna and H. E. Stanley, “Stochastic Process with Ultraslow Convergence to a Gaussian: The Truncated
        Levy Flight,” Phys. Rev. Lett., vol. 73, no. 22, pp. 2946–2949, 1994.

    :param size: optional
        Size can be a tuple with all the dimensions. Behaviour similar to ``numpy.random.standard_normal``.
    :param float beta: optional.
        Levy distribution parameter. The default is 1.5.

    :return: numpy.array
    """
    # Calculate x's std dev (Mantegna's algorithm)
    sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (
            np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)

    # Determine x and y using normal distributions with sigma_y = 1
    x = sigma * np.random.standard_normal(size)
    y = np.abs(np.random.standard_normal(size))
    z = np.random.standard_normal(size)

    # Calculate the random number with levy stable distribution
    return z * x / (y ** (1 / beta))

def random_search(pop, scale=0.01, distribution='uniform'):
    """
    Apply the random search from Random Search (RS) to the population's positions (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float scale: optional.
        It is the step scale. The default is 0.01.
    :param str distribution: optional.
        It is the distribution used to perform the random search. The default is 'uniform'.

    :return: None.
    """
    # Determine the random step
    if distribution == "uniform":
        random_step = np.random.uniform(-1, 1, (pop.num_agents, pop.num_dimensions))
    elif distribution == "gaussian":
        random_step = np.random.standard_normal((pop.num_agents, pop.num_dimensions))
    elif distribution == "levy":
        random_step = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')

    # Move each agent using uniform random displacements
    pop.positions += scale * random_step

