# Name: Multi-Operator Hybrid Metaheuristic (MOHM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Multi-Operator Hybrid Metaheuristic (MOHM) combines the strengths of two different optimization operators: swarm_dynamic and spiral_dynamic. The swarm_dynamic operator is effective in exploring large solution spaces and can quickly find good solutions, while the spiral_dynamic operator is designed for fine-tuning and refining the search around promising regions.
# By using both operators, MOHM aims to balance exploration and exploitation, ensuring a thorough search of the solution space. This approach has been shown to perform well on various optimization problems, including the Rastrigin function, which is known for its large and flat landscape with multiple local minima.
# The use of the 'greedy' selector for swarm_dynamic ensures that promising solutions are quickly accepted, while the 'probabilistic' selector for spiral_dynamic allows for more exploratory moves around the current solution, promoting diversification. This combination helps MOHM to escape from local optima and potentially find the global minimum.
# The multi-run experiment provides insights into the robustness and reliability of the metaheuristic. By analyzing the fitness results across multiple executions, we can evaluate the average performance, convergence speed, and consistency of MOHM on the Rastrigin function. This approach helps in understanding how well the algorithm performs in practice and informs potential improvements or refinements to the metaheuristic design.