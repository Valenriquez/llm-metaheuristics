```python
import optuna

# Define the hyperparameter search space
def objective(trial):
    # Define the metaheuristic parameters
    heuristic_parameters = [
        ('random_search', {
            'scale': trial.suggest_float('scale', 0.01, 1.0),
            'distribution': 'selected_distribution',
        }, 'selected_selector'),
        ('central_force_dynamic', {
            'gravity': trial.suggest_float('gravity', 0.001, 0.1),
            'alpha': trial.suggest_float('alpha', 0.01, 0.1),
            'beta': trial.suggest_float('beta', 1.0, 2.0),
            'dt': trial.suggest_float('dt', 0.01, 0.1)
        }, 'selected_selector'),
        ('differential_mutation', {
            'expression': trial.suggest_categorical('expression', ['rand', 'best', 'current', 'current-to-best', 'rand-to-best', 'rand-to-best-and-current']),
            'num_rands': trial.suggest_int('num_rands', 1, 3),
            'factor': trial.suggest_float('factor', 0.1, 1.0)
        }, 'all'),
        ('genetic_crossover', {
            'pairing': 'selected_pairing',
            'crossover': 'selected_crossover',
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)
        }, 'all'),
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.4, 0.9),
            'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
            'distribution': 'selected_distribution',
        }, 'selected_selector'),
    ]

    # Load the benchmark function
    fun = bf.{self.benchmark_function}({self.dimensions})
    prob = fun.get_formatted_problem()

    # Evaluate the metaheuristic performance
    performance = evaluate_sequence_performance(heuristic_parameters, prob, num_agents=50, num_iterations=100, num_replicas=30)

    # Return the performance as the objective value
    return performance

# Create an Optuna study object
study = optuna.create_study(direction="minimize")

# Optimize the hyperparameters
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and performance
print("Best Hyperparameters:", study.best_params)
print("Best Performance:", study.best_value)
```