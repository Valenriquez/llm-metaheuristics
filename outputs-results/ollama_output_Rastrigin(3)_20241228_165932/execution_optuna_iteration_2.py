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
                'angle': trial.suggest_float('angle', 20.0, 25.0),
                'sigma': trial.suggest_float('sigma', 0.01, 0.1)
            },
            'probabilistic'
        ),
        (
            'swarm_dynamic',
            {
                'factor': trial.suggest_float('factor', 0.1, 0.9),
                'self_conf': trial.suggest_float('self_conf', 2.5, 3.0),
                'swarm_conf': trial.suggest_float('swarm_conf', 2.5, 3.0),
                'version': trial.suggest_categorical('version', ['inertial', 'constriction']),
                'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
            },
            'metropolis'
        ), 
    ]
    fun = bf.Rastrigin(3)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance