# Name: SwarmMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7631114523645598,
            'self_conf': 2.998221085685541,
            'swarm_conf': 2.7311927653003547,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
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
# The SwarmMetaheuristic uses a swarm_dynamic operator with specified parameters to solve the Rastrigin function. This approach aims to leverage the collective behavior of particles in a population to explore the solution space efficiently. The greedy selector helps in choosing the best solutions at each iteration, ensuring that the algorithm converges to an optimal or near-optimal solution.