```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import  population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

import optuna

def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    # ... Code remains the same ...

def objective(trial):
    # Define the sequence of operators
    heur = [
        ( # Search operator 1
        'gravitational_search',
        { 
        'gravity': trial.suggest_float('scale', 0.01, 1.0),
        'alpha': trial.suggest_float('scale', 0.01, 1.0),
    },
    'all'
    ),
    (   # Search operator 2
    'random_flight',
    {
        'scale': trial.suggest_float('scale', 0.01, 1.0),
        'distribution': 'levy',
        'beta': trial.suggest_float('scale', 0.01, 2.0),
    },
    'probabilistic'
    )
]
    
    # Define the problem
    fun = bf.HappyCat(30)
    prob = fun.get_formatted_problem()

    # Evaluate the performance of the sequence
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    # Return the performance metric
    return performance

# Create an Optuna study
study = optuna.create_study(direction="minimize")

# Optimize the hyperparameters
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and performance
print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)
```