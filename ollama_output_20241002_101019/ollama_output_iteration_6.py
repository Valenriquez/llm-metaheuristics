 # Name: CustomSwarmDynamicMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh


fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'operator': 'swarm_dynamic',
        'parameters': {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'selector': 'greedy'
    }
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named CustomSwarmDynamicMetaheuristic, derived from the given template and aiming to optimize a benchmark function using a dynamic swarm optimization approach with specific parameters for exploration and exploitation based on inertial version and uniform distribution. The selector used is greedy, which helps in navigating through the search space efficiently by prioritizing promising solutions. This combination of parameters and operators is chosen to leverage both global and local search capabilities effectively, ensuring a balance between exploration (wide sampling) and exploitation (deep probing).
# 1. The metaheuristic name reflects its use of swarm dynamics with dynamic behavior, allowing for flexibility in parameter settings.
# 2. The benchmark function Rastrigin(2) is selected due to its nonconvex nature, which is suitable for testing global optimization algorithms.
# 3. The swarm_dynamic operator is configured with parameters factor (0.7), self_conf (2.54), swarm_conf (2.56), version ('inertial'), and distribution ('uniform') to encourage a balance between exploration and exploitation during search.
# 4. The greedy selector is chosen for its effectiveness in optimizing complex, nonconvex problems by focusing on improving promising solutions while discarding less useful ones early.