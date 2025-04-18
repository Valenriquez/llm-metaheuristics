```python
{
    "operators": [
        {
            "operator": "central_force_dynamic",
            "parameters": {
                "gravity": 0.01,
                "alpha": 0.1,
                "beta": 2.0,
                "dt": 0.5
            },
            "selector": "probabilistic"
        },
        {
            "operator": "differential_mutation",
            "parameters": {
                "expression": "rand-to-best-and-current",
                "num_rands": 2,
                "factor": 1.5
            },
            "selector": "metropolis"
        },
        {
            "operator": "genetic_crossover",
            "parameters": {
                "pairing": "cost",
                "crossover": "blend",
                "mating_pool_factor": 0.5
            },
            "selector": "greedy"
        },
        {
            "operator": "genetic_mutation",
            "parameters": {
                "scale": 1.2,
                "elite_rate": 0.05,
                "mutation_rate": 0.3,
                "distribution": "gaussian"
            },
            "selector": "probabilistic"
        }
    ]
}
```