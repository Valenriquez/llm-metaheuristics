 To integrate Optuna for optimizing the hyperparameters of your metaheuristic, we need to define an objective function that Optuna can use to evaluate and optimize the performance metrics based on the suggestions provided by `trial.suggest_float` or similar functions depending on the type of hyperparameter you are tuning. Below is the modified version of your code with Optuna for optimizing hyperparameters:

```python
import optuna
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import metaheuristic as mh
import benchmark_functions as bf

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
            'crossover': trial.suggest_categorical('crossover', ['uniform', 'gaussian']), 
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
        }, 'all'),
        
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.4, 0.9),
            'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
            'version': trial.suggest_categorical('version', ['inertial', 'constriction']), 
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
        }, 'all')
    ]

    fun = bf.Rastrigin(30)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)

def hello():
    print("hello")
```

### Key Changes and Additions:
1. **Optuna Integration**: The `objective` function is defined where Optuna's `trial` object is used to suggest hyperparameters for both the genetic crossover and swarm dynamic operators. These are suggested using `trial.suggest_float`, `trial.suggest_categorical`, etc., depending on whether a float, integer, or categorical hyperparameter is being tuned.
2. **Categorical Parameters**: For parameters that can take on specific discrete values (like 'crossover' type and 'version' of the swarm dynamic), `trial.suggest_categorical` is used to select from predefined lists.
3. **Performance Metric**: The performance metric remains the same, calculating the median fitness plus the interquartile range as a robust measure of central tendency in the presence of outliers.
4. **Study and Optimization**: An Optuna study is created and optimized based on the `objective` function over 50 trials. After optimization, the best parameters found are printed along with the best performance metric value.

The added "hello" function at the end simply prints "hello" as a test or placeholder function to demonstrate how additional functions can be included in the script without affecting the main logic.