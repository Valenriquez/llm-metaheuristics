# Name: Enhanced Hybrid Metaheuristic with Probabilistic Operators and Optuna Optimization

# Code:
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
    # Optimize parameters of each operator
    operator1_param1 = trial.suggest_float('operator1_param1', 0.1, 0.9)
    operator1_param2 = trial.suggest_int('operator1_param2', 1, 10)

    operator2_param1 = trial.suggest_float('operator2_param1', 0.2, 0.8)
    operator2_param2 = trial.suggest_int('operator2_param2', 5, 20)

    # Define the sequence of operators
    heur = [
        (  # Search operator 1
            'Operator1',
            {
                'parameter1': operator1_param1,
                'parameter2': operator1_param2,
            },
            'Selector1'
        ),
        (  # Search operator 2
            'Operator2',
            {
                'parameter1': operator2_param1,
                'parameter2': operator2_param2,
            },
            'Selector2'
        )
    ]

    fun = bf.Rastrigin(2) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

# Run the optimization
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

# Print the best hyperparameters and performance
print("Best hyperparameters:", study.best_params)
print("Best performance:", study.best_value)