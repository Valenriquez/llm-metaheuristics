```python
# This is the optuna template:

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
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
        ('random_flight', {
            'scale': trial.suggest_float('scale', 0.1, 5.0),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian']),
            'beta': trial.suggest_float('beta', 1.0, 3.0)
        }, trial.suggest_categorical('selector', ['probabilistic', 'greedy'])),
        ('spiral_dynamic', {
            'radius': trial.suggest_float('radius', 0.1, 0.9),
            'angle': trial.suggest_float('angle', 10.0, 25.0),
            'sigma': trial.suggest_float('sigma', 0.01, 0.3)
        }, trial.suggest_categorical('selector', ['all', 'greedy', 'probabilistic'])),
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.1, 2.0),
            'self_conf': trial.suggest_float('self_conf', 0.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 0.5, 3.0),
            'version': trial.suggest_categorical('version', ['constriction']),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
        }, trial.suggest_categorical('selector', ['greedy', 'probabilistic', 'all'])),
        ('local_random_walk', {
            'probability': trial.suggest_float('probability', 0.25, 0.75),
            'scale': trial.suggest_float('scale', 0.1, 5.0),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
        }, trial.suggest_categorical('selector', ['probabilistic', 'greedy'])),
        ('swarm_dynamic', {
            'factor': trial.suggest_float('factor', 0.1, 2.0),
            'self_conf': trial.suggest_float('self_conf', 0.5, 3.0),
            'swarm_conf': trial.suggest_float('swarm_conf', 0.5, 3.0),
            'version': trial.suggest_categorical('version', ['inertial']),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian'])
        }, trial.suggest_categorical('selector', ['all', 'greedy', 'probabilistic']))
    ]
    
    fun = bf.Rastrigin(3)
    
    return evaluate_sequence_performance(heur, fun, num_agents=50, num_iterations=1000, num_replicas=10)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
