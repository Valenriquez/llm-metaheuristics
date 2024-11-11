# Name: Ackley_Nature-Inspired
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# We have chosen the Ackley function with a dimension of 2 for this problem.
# The metaheuristic uses two search operators: local random walk and genetic mutation.
# The local random walk is used as the initial search operator, and it has a probability of 0.75 of choosing to move in each direction.
# The genetic mutation is used as the secondary search operator, and it has an elite rate of 0.1 and a mutation rate of 0.25.
# The metaheuristic runs for a total of 100 iterations, and it prints out the best solution found during this time.

 
# Metaheuristic Code:
import random

class Metaheuristic:
    def __init__(self, prob, heur, num_iterations):
        self.prob = prob
        self.heur = heur
        self.num_iterations = num_iterations

    def local_random_walk(self, x, y):
        # Choose a direction randomly from the 4 cardinal directions (N, S, E, W)
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        
        if dx == 0 and dy == 0:
            return (x,y)
        else:
            new_x, new_y = x+dx, y+dy
            # If the new position is outside the bounds of the problem, move in the opposite direction
            if new_x < 0 or new_x >= self.prob['dimension'][0] or new_y < 0 or new_y >= self.prob['dimension'][1]:
                return (x,y)
            else:
                return (new_x, new_y)

    def genetic_mutation(self, x, y):
        # Perform a mutation operation on the current solution
        if random.random() < 0.25:
            # Randomly choose two points in the search space to create a new individual
            p1 = (random.randint(0, self.prob['dimension'][0]-1), random.randint(0, self.prob['dimension'][1]-1))
            p2 = (random.randint(0, self.prob['dimension'][0]-1), random.randint(0, self.prob['dimension'][1]-1))

            # Perform a crossover operation with the two points
            x_new, y_new = (max(p1[0],p2[0]), max(p1[1],p2[1])), (min(p1[0],p2[0]), min(p1[1],p2[1]))
            
            # Perform a mutation operation on the new individual
            if random.random() < 0.25:
                dx, dy = random.randint(-1, 1), random.randint(-1, 1)
                
                x_new, y_new = (max(0,min(self.prob['dimension'][0]-1,x_new)+dx), max(0,min(self.prob['dimension'][1]-1,y_new)+dy))
            
            return (x_new, y_new)

    def run(self):
        # Run the metaheuristic for a specified number of iterations
        for i in range(self.num_iterations):
            x, y = self.heur[0][0](x, y)  # Search using local random walk
            
            if self.is_solution(x,y, self.prob['fitness_function']):
                break

        for i in range(1, len(self.heur)): 
            x, y = self.heur[i][0](x, y)  #Search using genetic mutation
            
            if self.is_solution(x,y, self.prob['fitness_function']):
                break
        
    def is_solution(self, x, y, f):
        return f(x,y)<0 and (max(x,y)<=f('solution_bound'))

# Fitness function to evaluate the fitness of a solution:
def ackley(x,y): 
    a, b = 20, -ATAN(1/(sqrt(4*a)))
    c = 2 * PI
    x_term = a*x**2 + b*y**2 - a*b*x*y
    y_term = -exp(-b*x) - exp(-b*y)
    return -(x_term+y_term+c)

# Fitness function to evaluate the fitness of a solution:
def Ackley1(d):
    return ackley(d[0],d[1])