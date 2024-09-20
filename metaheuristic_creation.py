import benchmark_func as bf
import population as pp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



functions = bf.__all__
print(functions)

fun = bf.Rastrigin(2)
print(fun)

#fun.plot(samples=100, resolution=125)
  
samples = 100

space_constraints = np.array(fun.get_search_range()).T

X, Y = np.meshgrid(np.linspace(*space_constraints[0], samples),
                   np.linspace(*space_constraints[1], samples))
Z = np.array([fun.get_function_value([x, y])
          for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

"""  
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.5,
                antialiased=True, cmap=plt.cm.jet)
#plt.show()
"""

pop = pp.Population(fun.get_search_range(), num_agents=50)
pop.initialise_positions('vertex')
#print("Internal Positions: ", pop.positions)
#print("External Positions: ", pop.get_positions())

pop.evaluate_fitness(lambda x: fun.get_function_value(x))
print(pop.fitness)

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

