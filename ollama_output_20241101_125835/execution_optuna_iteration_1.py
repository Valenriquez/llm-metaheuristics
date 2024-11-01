## Name: Optuna-Enhanced Metaheuristic with Particle Swarm Optimization and Mutation Operators

**Code:**

```python
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import optuna
import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import  population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

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
        ("particle_swarm_optimization", {
            "operator": trial.suggest_categorical("operator_ps", ["best_worst_operator", "random_operator"]),
            "mutation_operator": trial.suggest_categorical("mutation_operator_ps", ["gaussian_mutation", "polynomial_mutation"]),
            "w": trial.suggest_float("w", 0.1, 1.0),
            "c1": trial.suggest_float("c1", 0.1, 2.0),
            "c2": trial.suggest_float("c2", 0.1, 2.0),
            "inertia_weight_decay": trial.suggest_float("inertia_weight_decay", 0.0, 1.0)
        }),
        ("mutation", {
            "operator": trial.suggest_categorical("operator_m", ["gaussian_mutation", "polynomial_mutation"]),
            "probability": trial.suggest_float("probability_m", 0.0, 1.0),
            "sigma": trial.suggest_float("sigma_m", 0.1, 1.0),
            "eta": trial.suggest_float("eta_m", 0.5, 2.0)
        })
    ]

    fun = bf.Rastrigin(2) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)
```