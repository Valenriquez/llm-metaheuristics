# Name: Advanced Adaptive Hybrid Metaheuristic (AHM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {
            'search_space': prob['bounds'],
            'num_samples': 20
        },
        'uniform'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'velocity_factor': 0.7,
            'inertia_weight': 0.9,
            'cognitive_coefficient': 1.4,
            'social_coefficient': 1.4,
            'num_particles': 50
        },
        'stochastic'
    ),
    (
        # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius_factor': 0.8,
            'angle_increment': 15,
            'sigma_factor': 0.2
        },
        'metropolis'
    )
]

# Convergence-enhancing strategies:
# 1. Early termination if convergence is detected based on fitness improvement.
# 2. Adaptive parameter tuning to fine-tune the operators during runtime.

convergence_threshold = 0.01
num_generations_without_improvement = 5

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # Uncomment to enable verbose output
# met.run() # Uncomment to run the metaheuristic

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    
    best_fitness_history = []
    consecutive_improvement_count = 0
    
    for gen in range(met.num_iterations):
        met.run_iteration()
        current_best_fitness = met.get_current_best_fitness()
        
        if not best_fitness_history:
            best_fitness_history.append(current_best_fitness)
        else:
            if abs(best_fitness_history[-1] - current_best_fitness) < convergence_threshold:
                consecutive_improvement_count += 1
            else:
                consecutive_improvement_count = 0
            
            best_fitness_history.append(current_best_fitness)
        
        # Early termination if no improvement in fitness for a certain number of generations
        if consecutive_improvement_count >= num_generations_without_improvement:
            print(f"Early termination: Converged at generation {gen}")
            break
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines a random sample, swarm dynamic, and spiral dynamic operator.
# The convergence-enhancing strategies include early termination if the fitness improvement falls below a threshold for a certain number of generations.
# Adaptive parameter tuning can be added to fine-tune each operator during runtime based on their performance.