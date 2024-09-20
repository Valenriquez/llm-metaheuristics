import numpy as np

class OperatorsError(Exception):
    pass


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

def swarm_dynamic(pop, factor=1.0, self_conf=2.54, swarm_conf=2.56, version='constriction', distribution='uniform'):
    """
    Apply the swarm dynamic from Particle Swarm Optimisation (PSO) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float factor: optional.
        Inertial or Kappa factor, depending of which PSO version is set. The default is 1.0.
    :param float self_conf: optional.
        Self confidence factor. The default is 2.54.
    :param float swarm_conf: optional.
        Swarm confidence factor. The default is 2.56.
    :param str version: optional.
        Version of the Particle Swarm Optimisation strategy. It can be 'constriction' or 'inertial'. The default is
        'constriction'.
    :param str distribution: optional.
        Distribution to draw the random numbers. It can be 'uniform', 'gaussian', and 'levy'.

    :return: None.
    """
    # Determine random numbers
    if distribution == 'uniform':
        r_1 = np.random.rand(pop.num_agents, pop.num_dimensions)
        r_2 = np.random.rand(pop.num_agents, pop.num_dimensions)
    elif distribution == 'gaussian':
        r_1 = np.random.randn(pop.num_agents, pop.num_dimensions)
        r_2 = np.random.randn(pop.num_agents, pop.num_dimensions)
    elif distribution == 'levy':
        r_1 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
        r_2 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')

    # Choose the PSO version = 'inertial' or 'constriction'
    if version == 'inertial':
        # Find new velocities
        pop.velocities = factor * pop.velocities + r_1 * self_conf * (
                pop.particular_best_positions - pop.positions) + \
                         r_2 * swarm_conf * (np.tile(pop.global_best_position, (pop.num_agents, 1)) - pop.positions)
    elif version == 'constriction':
        # Find the constriction factor chi using phi
        phi = self_conf + swarm_conf
        if phi > 4:
            chi = 2 * factor / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        else:
            chi = np.sqrt(factor)

        # Find new velocities
        pop.velocities = chi * (pop.velocities +
                                r_1 * self_conf * (pop.particular_best_positions - pop.positions) +
                                r_2 * swarm_conf * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                                                    pop.positions))
    else:
        raise OperatorsError('Invalid swarm_dynamic version')

    # Move each agent using velocity's information
    pop.positions += pop.velocities


 
