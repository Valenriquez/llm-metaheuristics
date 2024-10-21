**Python File for Hyperparameter Tuning Using Optuna**

```python
import optuna
import benchmarks as bf
from evaluate_sequence_performance import evaluate_sequence_performance

# Define the objective function for hyperparameter tuning
def objective(trial):
    # Define the hyperparameters to tune
    parameters_to_tune = [
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.4, 0.9),
            'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
            'version': 'selected_version',
            'distribution': 'selected_distribution'
        }),
        ('differential_mutation', {
            'expression': 'selected_expression',
            'num_rands': trial.suggest_int('num_rands', 1, 3),
            'factor': trial.suggest_float('factor', 0.1, 1.0)
        }),
        ('genetic_crossover', {
            'pairing': 'selected_pairing',
            'crossover': 'selected_crossover',
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)
        })
    ]

    # Create the heuristic object
    heur = Heuristic(parameters_to_tune)

    # Define the problem
    fun = bf.Rastrigin(2)
    prob = fun.get_formatted_problem()

    # Evaluate the performance of the heuristic
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    # Return the performance as the objective value
    return performance

# Create an Optuna study for hyperparameter tuning
study = optuna.create_study(direction="minimize")

# Run the hyperparameter tuning study
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and performance
print("Best Hyperparameters Found:")
print(study.best_params)

print("Best Performance Found:")
print(study.best_value)
```

**Explanation:**

* The code uses the `optuna` library for hyperparameter tuning.
* The `objective()` function defines the hyperparameters to tune and evaluates the performance of the heuristic with these hyperparameters.
* The `Heuristic()` class is assumed to be defined in the `optuna_builder` folder and encapsulates the hyperparameters and algorithms.
* The `evaluate_sequence_performance()` function is assumed to be defined in the `evaluate_sequence_performance` module and evaluates the performance of the heuristic on a given problem.
* The `Rastrigin()` problem is selected for hyperparameter tuning.
* The hyperparameter search is run for 50 trials.
* The best hyperparameters and performance are printed after the hyperparameter tuning process is complete.