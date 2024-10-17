 To integrate Optuna for optimizing the hyperparameters of your metaheuristic, you can follow the provided Python script where Optuna is used to optimize the parameters of the genetic crossover and swarm dynamics operators. Below, I'll guide you through enhancing the existing script with Optuna, ensuring it maintains its structure and logic while maximizing or minimizing the performance metric based on the problem definition (in this case, a Rastrigin function in 30 dimensions).

Here's how you can modify your code to include Optuna for hyperparameter optimization:

```python
import optuna
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import metaheuristic_library as mh
import benchmark_functions as bf

# Define the evaluation function using Optuna's framework
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

# Define the objective function for Optuna optimization
def objective(trial):
    heur = [
        ('genetic_crossover', {
            'pairing': 'tournament_2_100',  
            'crossover': trial.suggest_categorical('crossover', ['uniform']),  # Only uniform crossover supported
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
        }, 'all'),
        
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.4, 0.9),
            'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
            'version': trial.suggest_categorical('version', ['inertial']),  # Only inertial version supported
            'distribution': trial.suggest_categorical('distribution', ['uniform'])  # Only uniform distribution supported
        }, 'all')
    ]

    fun = bf.Rastrigin(30)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

# Create a study object and optimize the objective function
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)

# Add a function to print "hello"
def hello():
    print("hello")

# Call the hello function
hello()
```

This script integrates Optuna for optimizing the hyperparameters of your metaheuristic, ensuring that only specific categories and distributions are considered for each hyperparameter. The `objective` function now uses Optuna's API to suggest values for the hyperparameters, which will be optimized based on the performance metric defined in `evaluate_sequence_performance`. After optimization, it prints out the best parameters found and their corresponding performance metric.