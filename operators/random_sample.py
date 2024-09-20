import numpy as np

def random_sample(pop):
    """
    Apply the random_sample to the population's positions (pop.positions). This operator has no memory.

    :param population pop: population.
        It is a population object.

    :return: None.
    """
    # Create random positions using random numbers between -1 and 1
    pop.positions = np.random.uniform(-1, 1, (pop.num_agents, pop.num_dimensions))
