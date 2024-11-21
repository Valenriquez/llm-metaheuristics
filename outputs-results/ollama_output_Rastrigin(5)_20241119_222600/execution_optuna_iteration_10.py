```python
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import optuna
import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# Function to evaluate sequence performance
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

# Optuna objective function
def objective(trial):
    heur = [
        (
            'random_sample',
            {},
            'all'
        ),
        (
            'spiral_dynamic',
            {
                'radius': trial.suggest_float('spiral_radius', 0.1, 0.9), 
                'angle': trial.suggest_float('spiral_angle', 2, 25), 
                'sigma': trial.suggest_float('spiral_sigma', 0.01, 3)
            },
            'probabilistic'
        ),
        (
            'swarm_dynamic',
            {
                'factor': trial.suggest_float('swarm_factor', 0.01, 25), 
                'self_conf': trial.suggest_float('swarm_self_conf', 1, 3), 
                'swarm_conf': trial.suggest_float('swarm_swarm_conf', 1, 3),
                'version': 'inertial',
                'distribution': 'uniform'
            },
            'all'
        )
    ]
    
    fun = bf.Rastrigin(5)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

# Create study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)
```