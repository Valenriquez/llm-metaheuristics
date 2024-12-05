def objective(trial):
    heur = [
        # Modified Metaheuristic
        (
            'spiral_dynamic',
            {
                'radius': trial.suggest_float('radius', 0.1, 0.9),
                'angle': trial.suggest_float('angle', 22.5, 25),
                'sigma': trial.suggest_float('sigma', 0.01, 0.3)
            },
            'metropolis'
        ),
        (
            'random_sample',
            {},
            'probabilistic'
        ),
    ]