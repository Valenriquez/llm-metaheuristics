# Name: Adaptive Spiral Dynamics Metaheuristic (ASDM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.6740385933750622,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8877203050873893,
            'angle': 24.87642725970664,
            'sigma': 0.29940154960809545
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The ASDM combines the strengths of swarm dynamics and spiral dynamics to create a robust metaheuristic for optimization problems. Swarm dynamics allows for efficient exploration of the solution space by simulating the behavior of social animals, while spiral dynamics introduces a more systematic approach through the use of spirals, which can help in refining solutions around local minima. The combination of these two operators enables ASDM to efficiently explore and exploit the search space, making it suitable for a wide range of optimization problems.