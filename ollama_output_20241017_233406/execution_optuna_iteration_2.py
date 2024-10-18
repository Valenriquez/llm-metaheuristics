**Enhanced Metaheuristic Code with Optuna:**

```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh
import optuna

# Define the Rastrigin function
fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

def objective(trial):
    # Suggest hyperparameters using Optuna
    gravity = trial.suggest_float('gravity', 0.5, 2.0)
    alpha = trial.suggest_float('alpha', 0.01, 0.1)
    scale = trial.suggest_float('scale', 0.5, 2.0)
    beta = trial.suggest_float('beta', 1.0, 3.0)

    # Create the metaheuristic with the suggested hyperparameters
    heur = [
        ('gravitational_search', {'gravity': gravity, 'alpha': alpha}, 'metropolis'),
        ('random_flight', {'scale': scale, 'distribution': 'levy', 'beta': beta}, 'probabilistic')
    ]

    met = mh.Metaheuristic(prob, heur, num_iterations=100)
    met.verbose = True
    met.run()

    # Return the best fitness value
    _, f_best = met.get_solution()
    return f_best

# Create an Optuna study to optimize hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and fitness value
print("Best Hyperparameters:", study.best_params)
print("Best Fitness Value:", study.best_value)
```

**Explanation:**

* We import the `optuna` library for hyperparameter optimization.
* The `objective()` function defines the hyperparameter search space and performs a single run of the metaheuristic with the suggested parameters.
* We create an Optuna study with the objective function and specify the optimization direction as "minimize".
* We run the optimization with `n_trials=100`.
* Finally, we print the best hyperparameters and fitness value found by Optuna.

**Note:**

* The hyperparameter search space is based on the original code and can be adjusted as needed.
* The number of trials (`n_trials`) may need to be adjusted depending on the complexity of the optimization task.
* The best hyperparameters and fitness value obtained can be used to evaluate the metaheuristic performance further.