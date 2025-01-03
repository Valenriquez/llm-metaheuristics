def objective(trial):
    heur = [
(
    'random_sample',
    {},
    'greedy'
),
(
    'spiral_dynamic',
    {
        'radius': trial.suggest_float('radius', 0.1, 0.9),
        'angle': trial.suggest_float('angle', 5, 25),
        'sigma': trial.suggest_float('sigma', 0.01, 0.3)
    },
    'probabilistic'
),
(
    'swarm_dynamic',
    {
        'factor': trial.suggest_float('factor', 0.5, 1.0),
        'self_conf': trial.suggest_float('self_conf', 1.0, 3.0),
        'swarm_conf': trial.suggest_float('swarm_conf', 1.0, 3.0),
        'version': trial.suggest_categorical('version', ['inertial', 'constriction']),
        'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
    },
    'metropolis'
),
(
    'local_random_walk',
    {
        'probability': trial.suggest_float('probability', 0.5, 1.0),
        'scale': trial.suggest_float('scale', 0.1, 2.0),
        'distribution': trial.suggest_categorical('distribution', ['gaussian'])
    },
    'greedy'
),
    ]
    fun = bf.Rastrigin(3)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance