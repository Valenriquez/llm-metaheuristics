import numpy as np

class OperatorsError(Exception):
    pass

def genetic_crossover(pop, pairing='rank', crossover='blend', mating_pool_factor=0.4):
    """
    Apply the genetic crossover from Genetic Algorithm (GA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param str pairing: optional.
        It indicates which pairing scheme to employ. Pairing schemes available are: 'cost' (Roulette Wheel or
        Cost Weighting), 'rank' (Rank Weighting), 'tournament', 'random', and 'even-odd'.
        When tournament is chosen, tournament size (tp) and probability (tp) can be encoded such as
        'tournament_{ts}_{tp}', {ts} and {tp}. Writing only 'tournament' is similar to specify 'tournament_3_100'.
        The default is 'rank'.
    :param str crossover: optional.
        It indicates which crossover scheme to employ. Crossover schemes available are: 'single', 'two', 'uniform',
        'blend', and 'linear'. Likewise 'tournament' pairing, coefficients of 'linear' can be enconded such as
        'linear_{coeff1}_{coeff2}' where the offspring is determined as follows:
            ``offspring = coeff1 * father + coeff2 * mother``
        The default is 'blend'.
    :param float mating_pool_factor: optional.
        It indicates the proportion of population to disregard. The default is 0.4.

    :return: None.
    """
    # Mating pool size
    num_mates = int(np.round(mating_pool_factor * pop.num_agents))

    # Get parents (at least a couple per offspring)
    if len(pairing) > 10:  # if pairing = 'tournament_2_100', for example
        pairing, tournament_size, tournament_probability = pairing.split("_")
        tournament_size = int(tournament_size)
        if num_mates < tournament_size:
            num_mates = tournament_size
    else:  # dummy (it must not be used)
        tournament_size, tournament_probability = '3', '100'

    # Number of offsprings (or couples)
    num_couples = pop.num_agents - num_mates

    # Get the mating pool using the natural selection
    mating_pool_indices = np.argsort(pop.fitness)[:num_mates]
    #
    # Roulette Wheel (Cost Weighting) Selection
    if pairing == 'cost':
        # Cost normalisation from mating pool: cost-min(cost @ non mates)
        normalised_cost = pop.fitness[mating_pool_indices] - np.min(
            pop.fitness[np.setdiff1d(np.arange(pop.num_agents), mating_pool_indices)])

        # Determine the related probabilities
        probabilities = np.abs(normalised_cost / (np.sum(normalised_cost) + 1e-23))

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(np.cumsum(probabilities), np.random.rand(2 * num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))

    # Roulette Wheel (Rank Weighting) Selection
    elif pairing == 'rank':
        # Determine the probabilities
        probabilities = (mating_pool_indices.size - np.arange(
            mating_pool_indices.size)) / np.sum(np.arange(mating_pool_indices.size) + 1)

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(np.cumsum(probabilities), np.random.rand(2 * num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))

    # Tournament pairing
    elif pairing == 'tournament':
        # Calculate probabilities
        probability = float(tournament_probability) / 100.
        probabilities = probability * ((1 - probability) ** np.arange(tournament_size))

        # Initialise the mother and father indices
        couple_indices = np.full((2, num_couples), np.nan)

        # Perform tournaments until all mates are selected
        for couple in range(num_couples):
            mate = 0
            while mate < 2:
                # Choose tournament candidates
                random_indices = mating_pool_indices[np.random.permutation(mating_pool_indices.size)[:tournament_size]]

                # Determine the candidate fitness values
                candidates_indices = random_indices[np.argsort(pop.fitness[random_indices])]

                # Find the best according to its fitness and probability
                winner = candidates_indices[np.random.rand(tournament_size) < probabilities]
                if winner.size > 0:
                    couple_indices[mate, couple] = int(winner[0])
                    mate += 1

    # Random pairing
    elif pairing == 'random':
        # Return two random indices from mating pool
        couple_indices = mating_pool_indices[np.random.randint(mating_pool_indices.size, size=(2, num_couples))]

    # TODO: Check Even-and-Odd pairing
    # Even-and-Odd pairing
    # elif pairing == "even-odd":
    #     # Check if the num of mates is even
    #     mating_pool_size = mating_pool_indices.size - \
    #         (mating_pool_indices.size % 2)
    #     half_size = mating_pool_size // 2
    #
    #     # Dummy indices according to the mating pool size
    #     remaining = num_couples - half_size
    #     if remaining > 0:
    #         dummy_indices = np.tile(
    #             np.reshape(np.arange(mating_pool_size),
    #                        (-1, 2)).transpose(),
    #             (1, int(np.ceil(num_couples / half_size))))
    #     else:
    #         dummy_indices = np.reshape(np.arange(mating_pool_size),
    #                                    (-1, 2)).transpose()
    #
    #     # Return couple_indices
    #     couple_indices = mating_pool_indices[
    #         dummy_indices[:, :num_couples]]

    # If no pairing procedure recognised
    else:
        raise OperatorsError("Invalid pairing method")

    # Identify offspring indices
    offspring_indices = np.setdiff1d(np.arange(pop.num_agents), mating_pool_indices, True)

    # Prepare crossover variables
    if len(crossover) > 7:  # if crossover = 'linear_0.5_0.5', for example
        cr_split = crossover.split("_")
        if len(cr_split) == 1:
            crossover = cr_split
            coeff1 = coeff2 = 0.5
        elif len(cr_split) == 2:
            crossover = cr_split[0]
            coeff1 = coeff2 = cr_split[1]
        else:
            crossover = cr_split[0]
            coeff1 = cr_split[1]
            coeff2 = cr_split[2]
        coefficients = [float(coeff1), float(coeff2)]
    else:  # dummy (it must not be used)
        coefficients = [np.nan, np.nan]

    # Perform crossover and assign to population
    parent_indices = couple_indices.astype(np.int64)

    # Single-Point Crossover
    if crossover == 'single':
        # Determine the single point per each couple
        single_points = np.tile(np.random.randint(
            pop.num_dimensions, size=parent_indices.shape[1]), (pop.num_dimensions, 1)).transpose()

        # Crossover condition mask
        crossover_mask = np.tile(np.arange(pop.num_dimensions), (parent_indices.shape[1], 1)) <= single_points

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Two-Point Crossover
    elif crossover == 'two':
        # Find raw points
        raw_points = np.sort(np.random.randint(pop.num_dimensions, size=(parent_indices.shape[1], 2)))

        # Determine the single point per each couple
        points = [np.tile(raw_points[:, x], (pop.num_dimensions, 1)).transpose() for x in range(raw_points.shape[1])]

        # Range matrix
        dummy_matrix = np.tile(np.arange(pop.num_dimensions), (parent_indices.shape[1], 1))

        # Crossover condition mask (only for two points)
        crossover_mask = np.bitwise_or(dummy_matrix <= points[0], dummy_matrix > points[1])

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Uniform Crossover
    elif crossover == 'uniform':
        # Crossover condition mask (only for uniform crossover)
        crossover_mask = np.random.rand(parent_indices.shape[1], pop.num_dimensions) < 0.5

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Random blending crossover
    elif crossover == 'blend':
        # Initialise random numbers between 0 and 1
        beta_values = np.random.rand(parent_indices.shape[1], pop.num_dimensions)

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = beta_values * father_position + (1 - beta_values) * mother_position

    # Linear Crossover: offspring = coeff[0] * father + coeff[1] * mother
    elif crossover == 'linear':
        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = coefficients[0] * father_position + coefficients[1] * mother_position

    # If no crossover method recognised
    else:
        raise OperatorsError('Invalid pairing method')

    # Store offspring positions in the current population
    pop.positions[offspring_indices, :] = np.copy(offsprings)