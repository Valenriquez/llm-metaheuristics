def objective(trial):
    heur = [
        (
            'random_sample',
            {},
            'greedy'
        ),
        (
            'swarm_dynamic',
            {
                'factor': trial.suggest_float('factor', 0.1, 0.9),
                'self_conf': trial.suggest_float('self_conf', 2.1, 3.0),
                'swarm_conf': trial.suggest_float('swarm_conf', 2.1, 3.0),
                'version': trial.suggest_categorical('version', ['inertial', 'constriction']),
                'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
            },
            'probabilistic'
        ),
        (
            'local_random_walk',
            {
                'probability': trial.suggest_float('probability', 0.5, 0.9),
                'scale': trial.suggest_float('scale', 0.1, 1.5),
                'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
            },
            'metropolis'
        ),
    ]
    
    fun = bf.Rastrigin(3)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance