import numpy as np

class OperatorsError(Exception):
    pass

def random_flight(pop, scale=1.0, distribution='levy', beta=1.5):
    """
    Apply the random flight from Random Search (RS) to the population's positions (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float scale: optional.
        It is the step scale. The default is 1.0.
    :param str distribution: optional.
        It is the distribution to draw the random samples. The default is 'levy'.
    :param float beta: optional
        It is the distribution parameter between [1.0, 3.0]. This paramenter only has sense when distribution='levy'.
         The default is 1.5.

    :return: None.
    """

    # Get random samples
    if distribution == 'uniform':
        random_samples = np.random.uniform(
            size=(pop.num_agents, pop.num_dimensions))

    elif distribution == 'gaussian':
        # Normal with mu = 0 and sigma = parameter
        random_samples = np.random.standard_normal(
            (pop.num_agents, pop.num_dimensions))

    elif distribution == 'levy':
        # Calculate the random number with levy stable distribution
        random_samples = _random_levy(size=(pop.num_agents, pop.num_dimensions), beta=beta)

    else:
        raise OperatorsError('Invalid distribution!')

    # Move each agent using levy random displacements
    pop.positions += scale * random_samples * (pop.positions - np.tile(pop.global_best_position, (pop.num_agents, 1)))
