 To incorporate Optuna for hyperparameter tuning into your existing metaheuristic code, you can follow these steps. The provided code already has a good structure to be enhanced with Optuna, so we'll focus on integrating Optuna and adding the "hello" function at the end of the script.

Here is the complete version of your script with Optuna for hyperparameter tuning, including the "hello" function:

```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import optuna

# Define the objective function for Optuna
def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    num_cores = multiprocessing.cpu_count()
    results_parallel = Parallel(n_jobs=num_cores)(delayed(run_metaheuristic)() for _ in range(num_replicas))

    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

def objective(trial):
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

    fun = bf.Rastrigin(30)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

# Create a study and optimize the objective function using Optuna
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)

# Add a "hello" function at the end of the code that prints "hello"
def hello():
    print("hello")

# Call the hello function to print "hello"
hello()
```

This script does the following:
1. Defines an objective function for Optuna to optimize hyperparameters based on performance metrics.
2. Uses Optuna's `create_study` and `optimize` methods to find the best hyperparameters through trial-and-error.
3. Implements a "hello" function at the end of the script that prints "hello".

This approach ensures that the original structure and logic are maintained while leveraging Optuna for hyperparameter tuning, making your code more robust and efficient in finding optimal configurations for your metaheuristic algorithm.