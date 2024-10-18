```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

import optuna

# Define the objective function to be optimized
def objective(trial):
    # Define the metaheuristic sequence
    heur = [
        ('genetic_crossover', {
            'pairing': 'tournament_2_100',  
            'crossover': 'uniform', 
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
        }, 'all'),
        
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.4, 0.9),
            'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
            'version': 'inertial', 
            'distribution': 'uniform' 
        }, 'all')
    ]

    # Define the optimization problem
    fun = bf.Rastrigin(30)
    prob = fun.get_formatted_problem()

    # Evaluate the performance of the metaheuristic sequence
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    # Return the performance metric
    return performance

# Create an Optuna study to optimize the hyperparameters
study = optuna.create_study(direction="minimize")  # Minimize the performance metric
study.optimize(objective, n_trials=50)  # Run the optimization for 50 trials

# Print the best hyperparameters found
print("Best hyperparameters found:")
print(study.best_params)

# Print the best performance found
print("Best performance found:")
print(study.best_value)
```

**Explanation:**

* We add an `objective` function that defines the hyperparameter search space using Optuna's `suggest_*` methods.
* We optimize the hyperparameters using the `study.optimize()` method, specifying the `objective` function and the number of trials.
* The best hyperparameters and performance are printed using `study.best_params` and `study.best_value`, respectively.

**Note:**

* The `evaluate_sequence_performance()` function remains unchanged from the original code.
* The metaheuristic sequence and optimization problem are also kept the same.
* The hyperparameter search space can be modified as needed based on the specific metaheuristic and optimization problem.