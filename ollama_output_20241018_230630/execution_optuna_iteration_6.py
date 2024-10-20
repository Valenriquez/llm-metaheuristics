## Name: Optuna-Enhanced Metaheuristic for Metaheuristic Optimization

### Code:

```python
import sys
import optuna

def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    # ... Code remains the same ...

def objective(trial):
    # ... Code remains the same ...

    # Optimization of operator parameters
    for operator in sequence:
        if operator['type'] == 'operator_selected':
            operator['parameters']['variable_selected'] = trial.suggest_float(
                f'operator_{operator["name"]}_variable_selected', 0.01, 1.0)

    performance = evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas)
    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)
```

### Short explanation and justification:

* We have added code to optimize the parameters of each operator in the `sequence` using `trial.suggest_float`.
* The variable names used for hyperparameter optimization are based on the `operator_name_variable_selected` convention.
* The `evaluate_sequence_performance` function remains unchanged.

### Remember:

* This Optuna-enhanced metaheuristic follows the original structure and logic of the given metaheuristic.
* The best hyperparameters and performance metrics are printed after optimization.