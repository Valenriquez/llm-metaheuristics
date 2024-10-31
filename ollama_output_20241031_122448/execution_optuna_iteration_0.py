# Name: Swarm-Based Metaheuristic with Levy Distribution Enhanced by Optuna

# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)   
prob = fun.get_formatted_problem()

def objective(trial):
    factor = trial.suggest_float('factor', 0.5, 0.9)
    self_conf = trial.suggest_float('self_conf', 2.5, 3)
    swarm_conf = trial.suggest_float('swarm_conf', 2.5, 3)

    heur = [
        ('swarm_dynamic',
            {
                'factor': factor,
                'self_conf': self_conf,
                'swarm_conf': swarm_conf,
                'version': 'inertial',
                'distribution': 'levy'
            },
            'probabilistic'
        )
    ]

    met = mh.Metaheuristic(prob, heur, num_iterations=100)
    met.verbose = True
    met.run()

    return met.get_solution()[1]

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Best hyperparameters found:")
print(study.best_params)

print("Best performance found:")
print(study.best_value)