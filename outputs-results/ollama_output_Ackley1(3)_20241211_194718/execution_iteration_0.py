# Name: Ackley1_3D_SwarmDynamic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': "uniform"
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=4) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The swarm_dynamic operator is used with a low inertia weight (factor=0.7) to ensure exploration, while the local_random_walk operator adds randomness for diversification. This combination helps in efficiently exploring the search space of Ackley1(3), a benchmark function known for its deceptive nature.