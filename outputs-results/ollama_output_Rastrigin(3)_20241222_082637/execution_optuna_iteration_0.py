def objective(trial):
    heur = [
        (
            'random_search',
            {
                'scale': trial.suggest_float('scale', 0.01, 0.9),
                'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
            },
            'greedy'
        ),
        (
            'central_force_dynamic',
            {
                'gravity': trial.suggest_float('gravity', 0.001, 0.005),
                'alpha': trial.suggest_float('alpha', 0.01, 0.02),
                'beta': trial.suggest_categorical('beta', ['1.4', '1.6']),
                'dt': trial.suggest_float('dt', 0.8, 1.4)
            },
            'probabilistic'
        ),
        (
            'differential_mutation',
            {
                'expression': trial.suggest_categorical('expression', ['rand-to-best', 'rand-to-best-and-current']),
                'num_rands': trial.suggest_int('num_rands', 1, 2),
                'factor': trial.suggest_float('factor', 0.8, 1.0)
            },
            'metropolis'
        ),
        (
            'firefly_dynamic',
            {
                'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian']),
                'alpha': trial.suggest_float('alpha', 1.0, 1.3),
                'beta': trial.suggest_float('beta', 1.0, 1.2),
                'gamma': trial.suggest_int('gamma', 100, 150)
            },
            'all'
        ),
    ]

    fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance