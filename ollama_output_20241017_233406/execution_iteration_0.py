 # Name: GravitationalSearchAlgorithmMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        "gravitational_search",
        { 
            "gravity": 1.0,
            "alpha": 0.02
        },
        "all"
    ),
    (  
        "random_flight",
        {
            "scale": 1.0,
            "distribution": "levy",
            "beta": 1.5
        },
        "probabilistic"
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is Gravitational Search Algorithm (GSA), inspired by the laws of gravity and mass interaction. In this implementation, we use a gravitational search with parameters set to typical values for exploration in optimization tasks. The first operator is a local search using gravitational forces influenced by the problem's alpha parameter, aiming at convergence through interactions similar to physical masses. The second operator introduces random flight patterns characterized by scale and distribution type (in this case, levy distribution), which helps in exploring broader regions of the solution space, enhancing global exploration. Both operators use the "all" selector to apply their effects across all candidate solutions during each iteration.
