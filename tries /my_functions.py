import numpy as np
def sphere_function(x):
    """
    Sphere function: f(x) = sum(x_i^2)
    Global minimum at x = 0, f(x) = 0
    Typically used with bounds [-5.12, 5.12] for all x_i
    """
    return np.sum(np.square(x))
    # sum((x_i - 0)^2)
    # x_i = Each component of a vector x in an n-dimensional space.
    # (x_i - 0): This subtracts 0 from each component. In practice, this doesn't change the value of x_i, but it's written this way to show the general form of the function.
    # ^2: This squares each component.
    # sum(...): This sums up all the squared components, resulting in a single scalar value.
    # MATH: f(x) = Σ(x_i - 0)² for i = 1 to n

def separable_ellipsoidal_function(x):  
    """
    Separable Ellipsoidal Function: f(x) = sum(i * x_i^2)
    Global minimum at x = 0, f(x) = 0
    Typically used with bounds [-5.12, 5.12] for all x_i
    """
    return np.sum(np.arange(1, len(x) + 1) * np.square(x))
#  Introduces a linear scaling factor i for each dimension

def rastrigin_function(x):
    """
    Rastrigin Function: f(x) = 10n + sum(x_i^2 - 10cos(2πx_i))
    Global minimum at x = 0, f(x) = 0
    Typically used with bounds [-5.12, 5.12] for all x_i
    """
    n = len(x)
    return 10 * n + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
    # a non-convex function, many local minima


def buche_rastrigin_function(x):
    """
    Büche-Rastrigin Function
    Global minimum at x = 0, f(x) = 0
    Typically used with bounds [-5, 5] for all x_i
    """
    n = len(x)
    s = 0
    for i in range(n):
        if x[i] > 0:
            x[i] = 10 * x[i]
        s += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return 10 * n + s + 100 * np.sum(np.square(np.maximum(0, np.abs(x) - 5)))


def linear_slope_function(x, alpha=10):
    """
    Linear Slope Function
    Global minimum at x = [5, 5, ..., 5] for positive alpha
    Typically used with bounds [-5, 5] for all x_i
    """
    z = np.where(x * alpha > 0, 5, -5)
    return np.sum(alpha * np.abs(x - z))


# Example with random input 
# np.random.uniform(low, high, size) generates random numbers from a uniform distribution.
# dim: is the number of random numbers to generate, which determines the dimensionality of the vector.
dim = 10
x = np.random.uniform(-5.12, 5.12, dim)
result = sphere_function(x)
print(f"Sphere function: {result}")