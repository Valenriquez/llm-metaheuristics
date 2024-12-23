def objective(trial):
    heur = [
        ('random_search', 
         {
             'scale': trial.suggest_float('scale', 0.01, 0.9),
             'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
         },
         trial.suggest_categorical('selection', ['greedy', 'all'])
        ),
        ('central_force_dynamic', 
         {
             'gravity': trial.suggest_float('gravity', 0.001, 0.01),
             'alpha': trial.suggest_float('alpha', 0.01, 0.1),
             'beta': trial.suggest_categorical('beta', ['1.25', '1.5', '2']),
             'dt': trial.suggest_float('dt', 0.1, 1)
         },
         trial.suggest_categorical('neighborhood', ['all', 'best'])
        ),
        ('differential_mutation', 
         {
             'expression': trial.suggest_categorical('expression', ['rand', 'best', 'current']),
             'num_rands': trial.suggest_int('num_rands', 1, 3),
             'factor': trial.suggest_float('factor', 0.5, 2)
         },
         trial.suggest_categorical('acceptance', ['metropolis', 'simulated_annealing'])
        ),
        ('firefly_dynamic', 
         {
             'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian']),
             'alpha': trial.suggest_float('alpha', 0.5, 1.5),
             'beta': trial.suggest_float('beta', 0.5, 1.5),
             'gamma': trial.suggest_int('gamma', 100, 200)
         },
         trial.suggest_categorical('selection', ['probabilistic', 'elite'])
        )
    ]
    fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=10, num_iterations=100, num_replicas=30)

    return performance