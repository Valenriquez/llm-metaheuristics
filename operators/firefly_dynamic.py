import numpy as np

class OperatorsError(Exception):
    pass

def firefly_dynamic(pop, alpha=1.0, beta=1.0, gamma=100.0, distribution='uniform'):
    """
    Apply the firefly dynamic from Firefly algorithm (FA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float alpha: optional.
        Scale of the random value. The default is 1.0.
    :param float beta: optional.
        Scale of the firefly contribution. The default is 1.0.
    :param float gamma: optional.
        Light damping parameters. The default is 100.
    :param str distribution: optional.
        Type of random number. Possible options: 'gaussian', 'uniform', and 'levy'. The default is 'uniform'.

    :return: None.
    """
    # Determine epsilon values
    if distribution == 'gaussian':
        epsilon_value = np.random.standard_normal((pop.num_agents, pop.num_dimensions))

    elif distribution == 'uniform':
        epsilon_value = np.random.uniform(-0.5, 0.5, (pop.num_agents, pop.num_dimensions))
    elif distribution == 'levy':
        epsilon_value = _random_levy((pop.num_agents, pop.num_dimensions), 1.5)
    else:
        raise OperatorsError('Invalid distribution')

    # Initialise delta or difference between two positions
    difference_positions = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vector distances with respect to agent
        delta = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        # Determine differences between lights
        delta_lights = np.tile((pop.fitness[indices] - np.tile(
            pop.fitness[agent], (1, pop.num_agents - 1))).transpose(), (1, pop.num_dimensions))

        # Find the total attraction for each agent
        difference_positions[agent, :] = np.sum(np.heaviside(-delta_lights, 0.0) * delta * np.exp(-gamma * np.tile(
            np.linalg.norm(delta, 2, 1).reshape(pop.num_agents - 1, 1), (1, pop.num_dimensions)) ** 2), 0)

    # Move fireflies according to their attractions
    pop.positions += alpha * epsilon_value + beta * difference_positions

