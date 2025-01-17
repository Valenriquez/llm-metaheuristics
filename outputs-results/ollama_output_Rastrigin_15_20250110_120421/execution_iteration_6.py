# Name: Adaptive Hybrid Metaheuristic (AHM)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sampling
        'random_sampling',
        {
            'step_size': 0.1,
        },
        'greedy'
    ),
    (
        'swarm_dynamics',  # Search operator 2: Swarm Dynamics
        {
            'inertia_weight': 0.7,
            'cognitive_coefficient': 1.4,
            'social_coefficient': 1.4,
        },
        'metropolis'
    ),
    (
        'spiral_dynamics',  # Search operator 3: Spiral Dynamics
        {
            'radius': 2.0,
            'angle': 0.5,
            'sigma': 0.1,
        },
        'probabilistic'
    )
]

class AdaptiveHybridMetaheuristic(mh.Metaheuristic):
    def __init__(self, problem, heuristic_list, num_iterations, num_agents):
        super().__init__(problem, heuristic_list, num_iterations, num_agents)
        self.performance_history = []

    def run(self):
        for _ in range(self.num_iterations):
            fitness_values = []
            for agent in self.agents:
                if np.random.rand() < 0.5:  # Random Sampling
                    agent.position += (np.random.rand(len(agent.position)) - 0.5) * self.heuristic_list[0][1]['step_size']
                else:  # Swarm Dynamics and Spiral Dynamics
                    best_agent = min(self.agents, key=lambda x: x.fitness)
                    cognitive_component = self.heuristic_list[1][1]['cognitive_coefficient'] * np.random.rand(len(best_agent.position)) * (best_agent.position - agent.position)
                    social_component = self.heuristic_list[2][1]['social_coefficient'] * np.random.rand(len(best_agent.position)) * (np.mean([other.position for other in self.agents if other is not agent], axis=0) - agent.position)
                    spiral_component = self.heuristic_list[2][1]['radius'] * np.sin(self.heuristic_list[2][1]['angle']) + self.heuristic_list[2][1]['sigma'] * np.random.rand(len(best_agent.position))
                    velocity = (self.inertia_weight * agent.velocity) + cognitive_component + social_component + spiral_component
                    agent.position += velocity

                if agent.position.any() < problem.lower_bound or agent.position.any() > problem.upper_bound:
                    agent.position = self._repair_position(agent.position)

                fitness_values.append(self.objective_function(agent.position))

            min_fitness = min(fitness_values)
            self.performance_history.append(min_fitness)

            # Adaptive Population Size
            if len(self.performance_history) >= 10 and np.mean(self.performance_history[-5:]) < np.mean(self.performance_history[:-5]):
                self.num_agents += 2
            elif len(self.performance_history) >= 10 and np.mean(self.performance_history[-5:]) > np.mean(self.performance_history[:-5]):
                self.num_agents = max(1, self.num_agents - 2)

        # Dynamic Parameter Tuning
        if len(self.performance_history) >= 10:
            recent_performance = np.array(self.performance_history[-10:])
            if np.std(recent_performance) < 0.1:
                for i in range(len(self.heuristic_list)):
                    self.heuristic_list[i][1]['step_size'] *= 0.9
                    self.heuristic_list[i][1]['inertia_weight'] *= 0.95
                    self.heuristic_list[i][1]['cognitive_coefficient'] *= 0.95
                    self.heuristic_list[i][1]['social_coefficient'] *= 0.95

        super().run()

# Main execution
met = AdaptiveHybridMetaheuristic(prob, heur, num_iterations=1000, num_agents=20)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = AdaptiveHybridMetaheuristic(prob, heur, num_iterations=1000, num_agents=20)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, met.best_agent.position, met.best_agent.fitness))

    fitness.append(met.best_agent.fitness)

# Output the average and standard deviation of the best fitness values found
print("Average Best Fitness:", np.mean(fitness))
print("Standard Deviation of Best Fitness:", np.std(fitness))

# Dynamic Parameter Tuning
if len(fitness) >= 10:
    recent_fitness = np.array(fitness[-10:])
    if np.std(recent_fitness) < 0.1:
        for i in range(len(heur)):
            heur[i][1]['step_size'] *= 0.9
            heur[i][1]['inertia_weight'] *= 0.95
            heur[i][1]['cognitive_coefficient'] *= 0.95
            heur[i][1]['social_coefficient'] *= 0.95

# Main execution (continued)
met = AdaptiveHybridMetaheuristic(prob, heur, num_iterations=1000, num_agents=20)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register (continued)
fitness = []
# Run the metaheuristic with the same problem 30 times (continued)
for rep in range(30):
    met = AdaptiveHybridMetaheuristic(prob, heur, num_iterations=1000, num_agents=20)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, met.best_agent.position, met.best_agent.fitness))

    fitness.append(met.best_agent.fitness)

# Output the average and standard deviation of the best fitness values found (continued)
print("Average Best Fitness:", np.mean(fitness))
print("Standard Deviation of Best Fitness:", np.std(fitness))