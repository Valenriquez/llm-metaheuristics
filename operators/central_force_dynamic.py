import numpy as np

def central_force_dynamic(pop, gravity=0.001, alpha=0.01, beta=1.5, dt=1.0):
    """
    Apply the central force dynamic from Central Force Optimisation (CFO) to the population's positions (pop.positions).

    :param population pop: population.
    :param float gravity: optional.
        It is the gravitational constant. The default is 0.001.
    :param float alpha: optional.
        It is the power mass factor. The default is 0.01.
    :param float beta: optional.
        It is the power distance factor. The default is 1.5.
    :param float dt: optional.
        It is the time interval between steps. The default is 1.0.

    :return: None.
    """
    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all masses differences with respect to agent
        delta_masses = pop.fitness[indices] - np.tile(
            pop.fitness[agent], (1, pop.num_agents - 1))

        # Determine all vector distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        distances = np.linalg.norm(delta_positions, 2, 1)

        # Find the quotient part
        quotient = np.heaviside(-delta_masses, 0.0) * (np.abs(delta_masses) ** alpha) / (distances ** beta + 1e-23)

        # Determine the acceleration for each agent
        acceleration[agent, :] = gravity * np.sum(delta_positions * np.tile(
            quotient.transpose(), (1, pop.num_dimensions)), 0)

    pop.positions += 0.5 * acceleration * (dt ** 2)

