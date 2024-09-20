import numpy as np

class OperatorsError(Exception):
    pass


def local_random_walk(pop, probability=0.75, scale=1.0, distribution='uniform'):
    """
    Apply the local random walk from Cuckoo Search (CS) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float probability: optional.
        It is the probability of discovering an alien egg (change an agent's position). The default is 0.75.
    :param float scale: optional.
        It is the step scale. The default is 1.0.
    :param str distribution: optional.
        It is the random distribution used to sample the stochastic variable. The default value is 'uniform'.

    :return: None.
    """

    # Determine random numbers
    if distribution == "uniform":
        r_1 = np.random.rand(pop.num_agents, pop.num_dimensions)
    elif distribution == "gaussian":
        r_1 = np.random.randn(pop.num_agents, pop.num_dimensions)
    elif distribution == "levy":
        r_1 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')
    r_2 = np.random.rand(pop.num_agents, pop.num_dimensions)

    # Move positions with a displacement due permutations and probabilities
    pop.positions += scale * r_1 * (pop.positions[
                                    np.random.permutation(pop.num_agents), :] - pop.positions[
                                                                                np.random.permutation(pop.num_agents),
                                                                                :]) * np.heaviside(r_2 - probability,
                                                                                                   0.0)
