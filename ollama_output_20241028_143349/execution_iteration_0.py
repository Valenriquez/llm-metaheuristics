# Name: Random Search

# Code:
heur = [
    (
    "random_walk",
    {
        "scale": 1.0,
        "distribution": "uniform"
    },
    "greedy"
    )
]

problem_id=2
instance=1
dimension=2
num_agents=100
num_iterations=400
num_replicas=1

evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y

# Short explanation and justification:
# This algorithm is a simple and easy to implement approach to the optimization problem. The "random_walk" operator allows the agents to randomly explore the search space. The "greedy" selector ensures that the algorithm selects the most promising solutions. This approach does not require much computation time, making it a suitable option for complex problems with large dimensions.