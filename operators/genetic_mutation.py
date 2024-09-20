import numpy as np

class OperatorsError(Exception):
    pass

def genetic_mutation(pop, scale=1.0, elite_rate=0.1, mutation_rate=0.25, distribution='uniform'):
    """
    Apply the genetic mutation from Genetic Algorithm (GA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float scale: optional.
        It is the scale factor of the mutations. The default is 1.0.
    :param float elite_rate : optional.
        It is the proportion of population to preserve. The default is 0.1.
    :param float mutation_rate: optional.
        It is the proportion of population to mutate. The default is 0.25.
    :param str distribution: optional.
        It indicates the random distribution that power the mutation. There are only two distribution available:
        'uniform', 'gaussian', and 'levy'. The default is 'uniform'.
    :return: None.
    """

    # Calculate the number of elite agents
    num_elite = int(np.ceil(pop.num_agents * elite_rate))

    # If num_elite equals num_agents then do nothing, or ...
    if num_elite < pop.num_agents:
        # Number of mutations to perform
        num_mutations = int(np.round(pop.num_agents * pop.num_dimensions * mutation_rate))

        # Identify mutable agents
        dimension_indices = np.random.randint(0, pop.num_dimensions, num_mutations)

        if num_elite > 0:
            agent_indices = np.argsort(pop.fitness)[np.random.randint(num_elite, pop.num_agents, num_mutations)]
        else:
            agent_indices = np.random.randint(num_elite, pop.num_agents, num_mutations)

        flat_rows = agent_indices.flatten()
        flat_columns = dimension_indices.flatten()

        total_mutations = len(flat_rows)

        # Perform mutation according to the random distribution
        if distribution == 'uniform':
            mutants = np.random.uniform(-1, 1, total_mutations)

        elif distribution == 'gaussian':
            # Normal with mu = 0 and sigma = parameter
            mutants = np.random.standard_normal(total_mutations)

        elif distribution == 'levy':
            mutants = _random_levy(total_mutations, 1.5)

        else:
            raise OperatorsError('Invalid distribution!')

        # Store mutants
        pop.positions[flat_rows, flat_columns] += scale * mutants

