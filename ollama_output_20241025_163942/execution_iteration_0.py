# Name: HybridMetaheuristic
# Code:

heur = [
    (
        "swarm_dynamic",
        {
            "factor": 0.7,
            "self_conf": 2.54,
            "swarm_conf": 2.56,
            "version": "inertial",
            "distribution": "uniform"
        },
        "metropolis"
    ),
    (
        "spiral_dynamic",
        {
            "radius": 0.9,
            "angle": 22.5,
            "sigma": 0.1
        },
        "probabilistic"
    )
]

problem_id = 2
instance = 1
dimension = 2
num_agents = 100
num_iterations = 100
num_replicas = 1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# The "HybridMetaheuristic" combines the "swarm_dynamic" and "spiral_dynamic" operators to take advantage of their diverse strengths.
# The "swarm_dynamic" operator with a factor of 0.7 and uniform distribution is chosen for its exploration capabilities.
# The "spiral_dynamic" operator with a radius of 0.9, angle of 22.5, and sigma of 0.1 is chosen for its exploitation capabilities.
# The "metropolis" and "probabilistic" selectors are chosen for their adaptability to different scenarios.
# By combining these two operators, the algorithm can balance exploration and exploitation, allowing it to search the solution space effectively.