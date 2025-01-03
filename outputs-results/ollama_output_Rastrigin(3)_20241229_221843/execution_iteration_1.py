# Name: HybridMetaheuristic
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
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.41557392820782957,
            'self_conf': 2.250242919153478,
            'swarm_conf': 2.743475108133401,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7056023568073462,
            'angle': 0.8201310273799196,
            'sigma': 0.08307470149367797
        },
        'greedy'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines the Swarm Dynamic operator with the Spiral Dynamic operator to leverage their complementary strengths. The Swarm Dynamic operator is good for global exploration due to its inertia-based approach, while the Spiral Dynamic operator can refine solutions locally by utilizing a spiral motion pattern. Together, this should provide a robust search strategy for optimization problems. The use of different selectors (probabilistic and greedy) allows for flexible control over the acceptance of new solutions, enhancing the overall efficiency of the algorithm.