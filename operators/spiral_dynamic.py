import numpy as np
from itertools import combinations as _get_combinations

class OperatorsError(Exception):
    pass

def get_rotation_matrix(dimensions, angle=0.3927):
    """
    Determine the rotation matrix by multiplying all the rotation matrices for each combination of 2D planes.

    :param int dimensions:
        Number of dimensions. Only positive integers greater than one.
    :param float angle: optional.
        Rotation angle (in radians). The default is 0.3927 radians (or 22.5 degrees).

    :return: numpy.array
        Rotation matrix to use over the population positions.
    """
    # Initialise the rotation matrix
    rotation_matrix = np.eye(dimensions)

    # Find the combinations without repetitions
    planes = list(_get_combinations(range(dimensions), 2))

    # Create the rotation matrix
    for xy in range(len(planes)):
        # Read dimensions
        x, y = planes[xy]

        # (Re)-initialise a rotation matrix for each plane
        rotation_plane = np.eye(dimensions)

        # Assign corresponding values
        rotation_plane[[x, x, y, y], [x, y, x, y]] = [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]
        rotation_matrix = np.matmul(rotation_matrix, rotation_plane)

    return rotation_matrix

def spiral_dynamic(pop, radius=0.9, angle=22.5, sigma=0.1):
    """
    Apply the spiral dynamic from Stochastic Spiral Optimisation (SSO) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float radius: optional.
        It is the convergence rate. The default is 0.9.
    :param float angle: optional.
        Rotation angle (in degrees). The default is 22.5 (degrees).
    :param float sigma: optional.
        Variation of random radii. The default is 0.1.
        Note: if sigma equals 0.0, the operator corresponds to that from the Deterministic Spiral Algorithm.

    :return: None.
    """
    # Determine the rotation matrix
    rotation_matrix = get_rotation_matrix(pop.num_dimensions, np.deg2rad(angle))

    for agent in range(pop.num_agents):
        random_radii = np.random.uniform(radius - sigma, radius + sigma, pop.num_dimensions)
        # If random radii need to be constrained to [0, 1]:
        pop.positions[agent, :] = pop.global_best_position + random_radii * \
                                  np.matmul(rotation_matrix, (pop.positions[agent, :] - pop.global_best_position))

