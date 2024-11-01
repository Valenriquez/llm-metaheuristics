```
Name: EnhancedSwarmMetaheuristic

Code:
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

# Note: If a word is in the code do not remove it, but if a number is in the code, replace it with "trial.suggest_float('variable_name', 0.1, 0.9)"
def objective(trial):
    heur = [
        (  # Search operator 1
            'swarm_dynamic',
            {
                'factor': trial.suggest_float('factor', 0.5, 0.9),
                'self_conf': trial.suggest_float('self_conf', 2.0, 3.0),
                'swarm_conf': trial.suggest_float('swarm_conf', 2.0, 3.0),
                'version': 'inertial',
                'distribution': 'uniform'
            },
            'probabilistic'
        ),
        (
            'swarm_dynamic',
            {
                'factor': trial.suggest_float('factor', 0.1, 0.6),
                'self_conf': trial.suggest_float('self_conf', 2.5, 3.5),
                'swarm_conf': trial.suggest_float('swarm_conf', 2.5, 3.5),
                'version': 'inertial',
                'distribution': 'uniform'
            },
            'probabilistic'
        )
    ]
    fun = bf.Rastrigin(2) # This is the selected problem, the problem may vary depending on the case.
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

#  IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.