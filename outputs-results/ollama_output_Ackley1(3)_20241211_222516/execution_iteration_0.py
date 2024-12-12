# Name: Ackley1 Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.33974742590800333,
            'self_conf': 2.225210348400868,
            'swarm_conf': 2.9823972332908575,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 0.9226426747350651,
            'beta': 1.556610326082572,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)  # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Ackley1 Metaheuristic combines the strengths of swarm dynamic and random flight operators. 
# The swarm_dynamic operator is used to explore the solution space effectively, while the random_flight operator helps in escaping local optima. 
# This combination aims to enhance both exploration and exploitation capabilities of the metaheuristic.
