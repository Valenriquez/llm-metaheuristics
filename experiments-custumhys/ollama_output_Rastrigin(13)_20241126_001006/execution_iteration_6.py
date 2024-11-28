# Name: Custom Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial",
            'distribution': "uniform"
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7,
            'angle': 30,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1500, num_agents=5) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines swarm dynamic and spiral dynamic operators to explore the search space efficiently. The use of probabilistic and metropolis selectors ensures that the algorithm can escape local minima. The number of agents is adjusted based on the dimension size, enhancing exploration in higher dimensions.