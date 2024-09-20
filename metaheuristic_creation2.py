import benchmark_func as bf
import population as pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pkgutil
import operators as op
import metaheuristic as mh

"""
# Construct the correct file path
file_path = os.path.join('llm-metaheuristics', 'collections', 'default.txt')

# Check if file exists and read it
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    for i, x in enumerate(data.split("\n")[:-1]):
        print(i, x, sep=" :: ")
else:
    print(f"File not found at {file_path}")

#dop = [eval(x) for x in data.decode().split("\n")[:-1]]
#pd.DataFrame(dop, columns=['Perturbator', 'Parameters', 'Selector'])

# Check if file exists and read it
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    for i, x in enumerate(data.split("\n")[:-1]):
        print(i, x, sep=" :: ")

    dop = [eval(x) for x in data.split("\n")[:-1]]
    df = pd.DataFrame(dop, columns=['Perturbator', 'Parameters', 'Selector'])
    print(df)
else:
    print(f"File not found at {file_path}")
""" 

functions = bf.__all__
print(functions)

fun = bf.Rastrigin(2)
print(fun)

#fun.plot(samples=100, resolution=125)
"""
samples = 100

space_constraints = np.array(fun.get_search_range()).T

X, Y = np.meshgrid(np.linspace(*space_constraints[0], samples),
                   np.linspace(*space_constraints[1], samples))
Z = np.array([fun.get_function_value([x, y])
          for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

print("X: ", X)
print("Y: ", Y)
print("Z: ", Z)
"""  

"""  
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.5,
                antialiased=True, cmap=plt.cm.jet)
#plt.show()
"""

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
 

"""   
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the op_tutorial.txt file
file_path = os.path.join(current_dir, "collections", "op_tutorial.txt")

try:
    with open(file_path, "r") as file:
        auto_operators = file.read()

    print(f"Collection size: {len(auto_operators)}")
    print(np.random.choice(auto_operators.split('\n'), 5))
except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Current directory:", current_dir)
    print("Contents of current directory:", os.listdir(current_dir))
    print("Contents of 'collections' directory:", os.listdir(os.path.join(current_dir, "collections")))
 
"""  
pop.update_positions(level='population', selector='all')
pop.update_positions(level='particular', selector='all')
pop.update_positions(level='global', selector='greedy')

"""
#print("get_state: ", pop.get_state())

print("----------Operator selection -------------------")
print('Perturbadores: ', op.__all__)
print('Selectores: ', op.__selectors__)

"""  
for iteration in range(10):
    # exec(op_to_exect)
    op.central_force_dynamic(pop, alpha=1.0)

    pop.evaluate_fitness(lambda x: fun.get_function_value(x))
    pop.update_positions(level='population', selector='all')
    pop.update_positions(level='global', selector='greedy')

    #print(f"Iteration: {iteration + 1} ::", pop.get_state())
    #show_positions()
 """


print("----------Metaheuristic creation-------------------")
fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()
heur = [( # Search operator 1
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'all'  # Selector
), (  # Search operator 2
    'differential_crossover',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'all'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()    # With this, the code runs and shows the results
#print('x_best = {}, f_best = {}'.format(*met.get_solution()))
