**Enhanced Code with Optuna:**

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
    # Suggest hyperparameters
    gravity = trial.suggest_float('gravity', 0.4, 0.6)
    alpha = trial.suggest_float('alpha', 0.005, 0.015)
    scale = trial.suggest_float('scale', 0.6, 0.8)
    beta = trial.suggest_float('beta', 1.2, 1.8)

    # Create the metaheuristic object
    heur = [
        ('gravitational_search', {'gravity': gravity, 'alpha': alpha}, 'metropolis'),
        ('random_flight', {'scale': scale, 'distribution': 'levy', 'beta': beta}, 'probabilistic'),
    ]
    met = mh.Metaheuristic(prob, heur, num_iterations=200)

    # Run the metaheuristic and return the fitness value
    met.run()
    _, f_best = met.get_solution()
    return f_best

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and fitness value
print('Best Hyperparameters:', study.best_params)
print('Best Fitness:', study.best_value)
```

**Changes Made:**

* Added an `objective()` function to optimize the hyperparameters using Optuna.
* Suggested hyperparameters for `gravity`, `alpha`, `scale`, and `beta` within their respective ranges.
* Created an `Optuna` study with a minimization direction.
* Optimized the hyperparameters using `n_trials` (50).

**Note:**

* The hyperparameter values used in the original code are replaced with suggested values from Optuna.
* The metaheuristic object is created with the suggested hyperparameters.
* The optimization process is run using `n_trials` iterations.
* The best hyperparameters and fitness value are printed at the end.

**Conclusion:**

This enhanced code incorporates Optuna for hyperparameter tuning, optimizing the performance of the Gravitational Search Optimization (GSO) metaheuristic. The best hyperparameters and fitness value are printed, providing insights into the optimal settings for the metaheuristic.