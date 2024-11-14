# Name: Alpine_Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Alpine1(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {
            # add more parameters as needed
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# After running this metaheuristic generator, you should see a solution that gives you a much better fitness value than 0. 
# The idea is to increase the number of agents when the dimension increases, to make sure the search space is covered properly.
# Initialize the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    # if the 'f_best' value is not improving over 10 runs, then something is wrong with the algorithm
    if rep >= 9 and (rep + 1) < 30:
        print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
        fitness.append(met.historical['fitness'])

# The idea here is to plot 'f_best' as it improves over the number of repits.
# You can use matplotlib for this. 

import matplotlib.pyplot as plt
plt.plot(fitness)
plt.xlabel('Repetition')
plt.ylabel('Fitness')
plt.title('Fitness Improvement Over Repetitions')
plt.show()