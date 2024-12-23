def objective(trial):
    heur = [
        (
            'random_search',
            {
                'scale': trial.suggest_float('scale', 0.1, 0.9),
                'distribution': trial.suggest_categorical('distribution', ['uniform'])
            },
            'greedy'
        ),
        (
            'central_force_dynamic',
            {
                'gravity': trial.suggest_float('gravity', 0.01, 0.9),
                'alpha': trial.suggest_float('alpha', 0.02, 0.9),
                'beta': trial.suggest_categorical('beta', ['1.5']),
                'dt': trial.suggest_float('dt', 0.5, 0.9)
            },
            'all'
        ),
        (
            'differential_mutation',
            {
                'expression': trial.suggest_categorical('expression', ['rand-to-best']),
                'num_rands': trial.suggest_int('num_rands', 1, 3),
                'factor': trial.suggest_float('factor', 1.0, 2.0)
            },
            'probabilistic'
        ),
        (
            'swarm_dynamic',
            {
                'factor': trial.suggest_categorical('factor', ['0.8']),
                'self_conf': trial.suggest_categorical('self_conf', ['3.0']),
                'swarm_conf': trial.suggest_categorical('swarm_conf', ['2.5']),
                'version': trial.suggest_categorical('version', ['inertial']),
                'distribution': trial.suggest_categorical('distribution', ['gaussian'])
            },
            'metropolis'
        )
    ]
    fun = bf.Rastrigin(3)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance