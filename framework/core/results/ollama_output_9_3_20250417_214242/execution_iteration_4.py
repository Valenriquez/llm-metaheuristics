```json
{
  "metaheuristic": {
    "operators": [
      "random_search",
      "central_force_dynamic",
      "differential_mutation",
      "firefly_dynamic",
      "genetic_crossover",
      "genetic_mutation",
      "gravitational_search",
      "random_flight",
      "local_random_walk",
      "swarm_dynamic"
    ],
    "parameters": {
      "random_search": {
        "scale": 0.01,
        "distribution": "gaussian"
      },
      "central_force_dynamic": {
        "gravity": 0.002,
        "alpha": 0.02,
        "beta": 1.3,
        "dt": 0.8
      },
      "differential_mutation": {
        "expression": "rand-to-best",
        "num_rands": 2,
        "factor": 1.5
      },
      "firefly_dynamic": {
        "distribution": "gaussian",
        "alpha": 0.9,
        "beta": 0.8,
        "gamma": 120.0
      },
      "genetic_crossover": {
        "pairing": "random",
        "crossover": "uniform",
        "mating_pool_factor": 0.5
      },
      "genetic_mutation": {
        "scale": 0.8,
        "elite_rate": 0.15,
        "mutation_rate": 0.20,
        "distribution": "gaussian"
      },
      "gravitational_search": {
        "gravity": 1.2,
        "alpha": 0.025
      },
      "random_flight": {
        "scale": 0.9,
        "distribution": "uniform",
        "beta": 1.4
      },
      "local_random_walk": {
        "probability": 0.78,
        "scale": 0.95,
        "distribution": "gaussian"
      },
      "swarm_dynamic": {
        "factor": 0.65,
        "self_conf": 2.4,
        "swarm_conf": 2.6,
        "version": "inertial",
        "distribution": "uniform"
      }
    },
    "selectors": [
      "metropolis",
      "probabilistic"
    ]
  }
}
```