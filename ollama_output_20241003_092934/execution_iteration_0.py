 # Name: Particle Swarm Optimization with Adaptive Mutation
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

# Parameters for Particle Swarm Optimization are taken from parameters_to_take.txt
parameters = {
    "swarm_dynamic": {
        "factor": 0.7,
        "self_conf": 2.54,
        "swarm_conf": 2.56,
        "version": "inertial",
        "distribution": "uniform"
    }
}

# Heuristic operators are selected based on the parameters provided in parameters_to_take.txt
heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is Particle Swarm Optimization (PSO), which is a population-based algorithm inspired by the behavior of bird flocks or fish schools. PSO uses a swarm to search for solutions, where each particle represents a potential solution in the search space.

# PSO involves updating particles' positions based on their personal best and the swarm's best position found so far. The update includes contributions from cognitive (personal best) and social components (swarm's best), adjusted by inertia weights that control the impact of the previous velocities.

# In this implementation, we use a custom operator named 'swarm_dynamic' which is based on the PSO principles. The parameters for this operator are taken from `parameters_to_take.txt`. These include:
# - factor: A scaling factor to adjust the particle's velocity.
# - self_conf: Cognitive parameter that scales the contribution of the personal best position.
# - swarm_conf: Social parameter that scales the contribution of the global (swarm's) best position.
# - version: The type of PSO version used, in this case, 'inertial'.
# - distribution: The distribution method for initializing velocities or positions; here set to 'uniform'.

# The selector used is 'probabilistic', meaning that the operator will be applied with a certain probability during each iteration. This allows for flexibility in applying different operators based on their success rates observed through iterations, which can be further optimized by adjusting probabilities and other parameters as needed.

# PSO typically benefits from adaptive mutation mechanisms to prevent premature convergence or stagnation. However, since the provided template does not explicitly include a genetic crossover and mutation mechanism (which is typical in algorithms like Genetic Algorithms), the focus here is on adapting the swarm dynamics through parameter settings for better exploration of the search space and exploitation of promising areas.

# The justification for using PSO with these specific parameters is to leverage its ability to balance between global and local explorations, which is crucial for continuous optimization problems like the Sphere function used in this example.