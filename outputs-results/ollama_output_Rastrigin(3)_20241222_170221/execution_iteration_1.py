# Name: Adaptive Differential Evolution with Perturbations (ADEP)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Differential Evolution
        'differential_evolution',
        {
            'crossover_probability': 0.8,
            'scaling_factor': 0.5
        },
        'elitist'
    ),
    (
        # Search operator 2: Perturbation
        'gaussian_perturbation',
        {
            'std_dev': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.best_solution))
    
    fitness.append(met.historical_fits)

# Short explanation and justification:
# ADEP combines Differential Evolution (a population-based global optimization algorithm) with Gaussian perturbations.
# The differential_evolution operator helps in exploring the solution space effectively by performing mutations and crossovers,
# while the gaussian_perturbation operator provides a way to escape local optima and refine solutions. 
# This hybrid approach ensures a balance between exploration and exploitation, leading to better convergence and higher quality solutions.