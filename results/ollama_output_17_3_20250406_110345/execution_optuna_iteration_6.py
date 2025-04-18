```python
import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from P1 import P1

# Parámetros iniciales
problem_id = 17    
instance = 1
dimension = 3     
num_agents = 57    
num_iterations = 100
num_replicas = 10

# Definir la función para evaluar el rendimiento de la secuencia
def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    # Número de núcleos disponibles para paralelización
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores, prefer="threads")(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Calcular la métrica de rendimiento usando la mediana y el IQR
    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

# Definir el objetivo de Optuna
def objective(trial):
    heur = [
        ('central_force_dynamic', {
            'gravity': trial.suggest_float('gravity', 0.2, 0.6),
            'alpha': trial.suggest_float('alpha', 0.01, 0.03),
            'beta': trial.suggest_categorical('beta', ['2.5', '3.0', '3.5']),
            'dt': trial.suggest_float('dt', 0.8, 1.2)
        }, 'metropolis'),
        ('genetic_crossover', {
            'pairing': trial.suggest_categorical('pairing', ['rank', 'cost', 'random', 'tournament_2_100']),
            'crossover': trial.suggest_categorical('crossover', ['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5']),
            'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.3, 0.5)
        }, 'probabilistic'),
        ('genetic_mutation', {
            'scale': trial.suggest_categorical('scale', ['0.01', '0.015', '0.02']),
            'elite_rate': trial.suggest_float('elite_rate', 0.05, 0.15),
            'mutation_rate': trial.suggest_categorical('mutation_rate', ['0.2', '0.25', '0.3']),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy'])
        }, 'metropolis'),
        ('random_flight', {
            'scale': trial.suggest_float('scale', 0.8, 1.2),
            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian', 'levy']),
            'beta': trial.suggest_categorical('beta', ['1.5', '2.0', '2.5'])
        }, 'all')
    ]

    performance = evaluate_sequence_performance(heur, prob, num_agents=num_agents, num_iterations=num_iterations, num_replicas=num_replicas)

    return performance

# Configurar y ejecutar el estudio de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```