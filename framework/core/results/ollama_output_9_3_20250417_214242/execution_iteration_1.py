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
      "local_random_walk"
    ],
    "parameters": {
      "random_search": {
        "scale": 0.1,
        "distribution": "uniform"
      },
      "central_force_dynamic": {
        "gravity": 0.01,
        "alpha": 0.02,
        "beta": 1.0,
        "dt": 0.5
      },
      "differential_mutation": {
        "expression": "current-to-best",
        "num_rands": 2,
        "factor": 1.5
      },
      "firefly_dynamic": {
        "distribution": "gaussian",
        "alpha": 0.5,
        "beta": 0.5,
        "gamma": 50.0
      },
      "genetic_crossover": {
        "pairing": "cost",
        "crossover": "uniform",
        "mating_pool_factor": 0.6
      },
      "genetic_mutation": {
        "scale": 1.2,
        "elite_rate": 0.2,
        "mutation_rate": 0.3,
        "distribution": "gaussian"
      },
      "gravitational_search": {
        "gravity": 0.5,
        "alpha": 0.03
      },
      "random_flight": {
        "scale": 1.2,
        "distribution": "gaussian",
        "beta": 1.7
      },
      "local_random_walk": {
        "probability": 0.8,
        "scale": 1.1,
        "distribution": "uniform"
      }
    },
    "selector": "probabilistic"
  }
}
```