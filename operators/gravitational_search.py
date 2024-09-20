import numpy as np

def gravitational_search(pop, gravity=1.0, alpha=0.02):
    """
    Apply the gravitational search from Gravitational Search Algorithm (GSA) to the population's positions
    (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float gravity: optional.
        It is the initial gravitational value. The default is 1.0.
    :param float alpha: optional.
        It is the gravitational damping ratio. The default is 0.02.

    :return: None.
    """

    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    # Determine the gravitational constant
    gravitation = gravity * np.exp(- alpha * pop.iteration)

    # Determine mass for each agent
    raw_masses = (pop.fitness - np.tile(pop.current_worst_fitness, (1, pop.num_agents)))
    masses = (raw_masses / (np.sum(raw_masses) + 1e-23)).reshape(pop.num_agents)

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vector distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        quotient = masses[indices] / (np.linalg.norm(delta_positions, 2, 1) + 1e-23)

        # Force interaction
        force_interaction = gravitation * np.tile(
            quotient.reshape(pop.num_agents - 1, 1), (1, pop.num_dimensions)) * delta_positions

        # Acceleration
        acceleration[agent, :] = np.sum(np.random.rand(pop.num_agents - 1, pop.num_dimensions) * force_interaction, 0)

    # Update velocities
    # TODO: Add different random distributions
    pop.velocities = acceleration + np.random.rand(pop.num_agents, pop.num_dimensions) * pop.velocities

    # Update positions
    pop.positions += pop.velocities

