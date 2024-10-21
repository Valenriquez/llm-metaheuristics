## Metaheuristic with Hyperparameter Tuning using Optuna

**Objective:**
The objective is to enhance the metaheuristic code by incorporating Optuna for hyperparameter tuning. The goal is to find the best set of hyperparameters that optimize the performance of the metaheuristic algorithm.

**Code:**

```python
import optuna  # Import Optuna library

# Define the hyperparameter search space
def objective(trial):
    # Define the metaheuristic parameters to tune
    parameters_to_tune = {
        'num_iterations': trial.suggest_int('num_iterations', 50, 200),
        'population_size': trial.suggest_int('population_size', 50, 200),
        'crossover_rate': trial.suggest_float('crossover_rate', 0.5, 1.0),
        'mutation_rate': trial.suggest_float('mutation_rate', 0.05, 0.2)
    }

    # Create the metaheuristic object with the tuned parameters
    met = mh.Metaheuristic(prob, sequence, **parameters_to_tune)

    # Run the metaheuristic algorithm
    met.run()

    # Evaluate the performance of the metaheuristic solution
    fitness_value = met.get_solution()[1]

    # Return the fitness value for Optuna optimization
    return fitness_value

# Create an Optuna study for hyperparameter tuning
study = optuna.create_study(direction="maximize")  # Use maximization for performance metric

# Optimize the hyperparameters
study.optimize(objective, n_trials=100)  # Run 100 trials of hyperparameter optimization

# Print the best hyperparameters and performance
print("Best hyperparameters:", study.best_params)
print("Best performance:", study.best_value)
```

**Explanation:**

* The `objective()` function defines the hyperparameter search space. It uses Optuna's `suggest_*()` methods to propose different values for each hyperparameter.
* An Optuna study is created with the `direction="maximize"` argument, indicating that we want to maximize the performance metric.
* The `optimize()` method runs the hyperparameter optimization process for 100 trials.
* Finally, the best hyperparameters and performance are printed.

**Parameters in parameters_to_take.txt:**

```
num_iterations: 100
population_size: 100
crossover_rate: 0.7
mutation_rate: 0.1
```

**Note:**

* This code assumes that the necessary imports, variables (`prob` and `sequence`), and the `Metaheuristic` class are already defined in the script.
* The specific hyperparameters and their search spaces may need to be adjusted depending on the chosen metaheuristic and problem.
* The number of trials in the `optimize()` function can be further increased for more thorough optimization.