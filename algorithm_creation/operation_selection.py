import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pkgutil

import sys

#parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh

#functions = bf.__all__
#print(functions)

fun = bf.Rastrigin(2)
#print(fun)


pop = pp.Population(fun.get_search_range(), num_agents=50)
pop.initialise_positions('vertex')
#print("Internal Positions: ", pop.positions)
#print("External Positions: ", pop.get_positions())

pop.evaluate_fitness(lambda x: fun.get_function_value(x))
print(pop.fitness)


samples = 100

space_constraints = np.array(fun.get_search_range()).T

""" 
X, Y = np.meshgrid(np.linspace(*space_constraints[0], samples),
                   np.linspace(*space_constraints[1], samples))
Z = np.array([fun.get_function_value([x, y])
          for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)


def show_positions(azim=-90, elev=90):
    positions = np.array(pop.get_positions())
    fitness = pop.fitness

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d', )
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.5,
                    antialiased=True, cmap=plt.cm.winter, alpha=0.25)
    ax.scatter(positions[:,0], positions[:,1], pop.fitness,
               marker='o', color='red')
    ax.view_init(elev, azim)
    plt.show()

show_positions() 
 """


# An important part of the population is the update of the particular positions, which are used by some operators.
pop.update_positions(level='population', selector='all')
pop.update_positions(level='particular', selector='all')
pop.update_positions(level='global', selector='greedy')


#print("get_state: ", pop.get_state())
print("----------Operator selection -------------------")
print('Perturbadores: ', op.__all__)
print('Selectores: ', op.__selectors__)


for iteration in range(10):
    # exec(op_to_exect)
    op.central_force_dynamic(pop, alpha=1.0)

    pop.evaluate_fitness(lambda x: fun.get_function_value(x))
    pop.update_positions(level='population', selector='all')
    pop.update_positions(level='global', selector='greedy')

    #print(f"Iteration: {iteration + 1} ::", pop.get_state())
    #show_positions()
 

