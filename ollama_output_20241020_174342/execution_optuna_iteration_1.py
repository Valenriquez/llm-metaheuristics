## Name: Enhanced Metaheuristic with Optuna

## Code:

```python
import optuna
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import population as pp
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
    # Heuristic sequence with hyperparameters to optimize
    heur = [
        trial.suggest_categorical('operator1', ['random_flight', 'local_random_walk', 'spiral_dynamic']),
        trial.suggest_float('operator2_scale', 0.1, 2.0),
        trial.suggest_categorical('operator3', ['swarm_dynamic']),
        trial.suggest_float('operator3_factor', 0.7, 1.0)
    ]

    fun = bf.Rastrigin(2)  # Replace with the selected problem
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters found:")
    print(study.best_params)

    print("Best performance found:")
    print(study.best_value)
```

**Note:**

* The `objective` function defines the optimization problem by proposing a heuristic sequence with hyperparameters as variables.
* The `evaluate_sequence_performance` function evaluates the performance of the proposed sequence using the metaheuristic with the specified hyperparameters.
* The hyperparameters to optimize can be modified based on the specific metaheuristic and problem.
* The selected problem is the `Rastrigin` function, but this can be replaced with any other problem from the `benchmark_func` library.