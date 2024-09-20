import numpy as np

class OperatorsError(Exception):
    pass

def differential_mutation(pop, expression='current-to-best', num_rands=1, factor=1.0):
    """
    Apply the differential mutation from Differential Evolution (DE) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param str expression: optional.
        Type of DE mutation. Available mutations: 'rand', 'best', 'current', 'current-to-best', 'rand-to-best',
         'rand-to-best-and-current'. The default is 'current-to-best'.
    :param int num_rands: optional.
        Number of differences between positions selected at random. The default is 1.
    :param float factor: optional.
        Scale factor (F) to weight contributions from other agents. The default is 1.0.

    :return: None.

    """
    # TODO: Include the expression 'current-to-pbest'
    # Create mutants using the expression provided in scheme
    if expression == 'rand':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :]

    elif expression == 'best':
        mutant = np.tile(pop.global_best_position, (pop.num_agents, 1))

    elif expression == 'current':
        mutant = pop.positions

    elif expression == 'current-to-best':
        mutant = pop.positions + factor * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                                           pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == 'rand-to-best':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :] + factor * (np.tile(
            pop.global_best_position, (pop.num_agents, 1)) - pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == 'rand-to-best-and-current':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :] + factor * (
                np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                pop.positions[np.random.permutation(pop.num_agents), :] +
                pop.positions[np.random.permutation(pop.num_agents), :] - pop.positions)
    else:
        raise OperatorsError('Invalid DE mutation scheme!')

    # Add random parts according to num_rands
    if num_rands >= 0:
        for _ in range(num_rands):
            mutant += factor * (pop.positions[np.random.permutation(pop.num_agents), :] -
                                pop.positions[np.random.permutation(pop.num_agents), :])
    else:
        raise OperatorsError('Invalid DE mutation scheme!')

    # Replace mutant population in the current one
    pop.positions = np.copy(mutant)

