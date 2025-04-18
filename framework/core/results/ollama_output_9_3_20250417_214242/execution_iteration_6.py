```json
{
  "metaheuristic": {
    "operators": [
      {
        "operator": "central_force_dynamic",
        "parameters": {
          "gravity": 0.01,
          "alpha": 0.02,
          "beta": 1.8,
          "dt": 0.5
        },
        "selector": "metropolis"
      },
      {
        "operator": "differential_mutation",
        "parameters": {
          "expression": "rand-to-best-and-current",
          "num_rands": 2,
          "factor": 1.2
        },
        "selector": "probabilistic"
      },
      {
        "operator": "genetic_crossover",
        "parameters": {
          "pairing": "cost",
          "crossover": "uniform",
          "mating_pool_factor": 0.5
        },
        "selector": "greedy"
      },
      {
        "operator": "genetic_mutation",
        "parameters": {
          "scale": 1.2,
          "elite_rate": 0.15,
          "mutation_rate": 0.3,
          "distribution": "gaussian"
        },
        "selector": "metropolis"
      },
      {
        "operator": "spiral_dynamic",
        "parameters": {
          "radius": 0.8,
          "angle": 25,
          "sigma": 0.15
        },
        "selector": "probabilistic"
      }
    ],
    "iteration_count": 6
  }
}
```