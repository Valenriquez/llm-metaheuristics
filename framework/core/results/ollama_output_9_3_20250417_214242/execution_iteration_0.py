```json
{
  "operators": {
    "central_force_dynamic": {
      "parameters": {
        "gravity": 0.01,
        "alpha": 0.02,
        "beta": 1.5,
        "dt": 0.5
      },
      "selector": "probabilistic"
    },
    "genetic_crossover": {
      "parameters": {
        "pairing": "cost",
        "crossover": "two",
        "mating_pool_factor": 0.3
      },
      "selector": "all"
    },
    "genetic_mutation": {
      "parameters": {
        "scale": 0.5,
        "elite_rate": 0.15,
        "mutation_rate": 0.2,
        "distribution": "gaussian"
      },
      "selector": "probabilistic"
    },
    "gravitational_search": {
      "parameters": {
        "gravity": 0.9,
        "alpha": 0.03
      },
      "selector": "metropolis"
    }
  }
}
```