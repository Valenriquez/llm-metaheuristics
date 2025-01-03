# Name: Multi-Strategy Hybrid Metaheuristic (MSHM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Flight with Gaussian distribution and Probabilistic selector
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'gaussian',
            'beta': 1.5
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic', # Search operator 2: Spiral Dynamic with Gaussian distribution and All selector
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'swarm_dynamic', # Search operator 3: Swarm Dynamic with Constriction version and Greedy selector
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'local_random_walk', # Search operator 4: Local Random Walk with Gaussian distribution and Probabilistic selector
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic', # Search operator 5: Swarm Dynamic with Inertial version and All selector
        {
            'factor': 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'all'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# MSHM combines five different search operators with varying characteristics to explore the solution space more thoroughly.
# The Random Flight operator helps in exploring the landscape by moving agents randomly but focusing on Gaussian distribution for better exploration.
# The Spiral Dynamic operator is designed to spiral outwards, providing a systematic exploration strategy.
# The Swarm Dynamic operator utilizes both self- and swarm confidences with different versions (Constriction and Inertial) to balance between exploitation and exploration.
# Each operator has been paired with a selector that determines when it should be used based on the search context, enhancing the overall efficiency of the metaheuristic.