```python
{
    "central_force_dynamic": {
        "parameters": {
            "gravity": 0.01,
            "alpha": 0.02,
            "beta": 2.0,
            "dt": 0.5
        },
        "selector": "metropolis"
    },
    "differential_mutation": {
        "parameters": {
            "expression": "rand-to-best",
            "num_rands": 2,
            "factor": 1.2
        },
        "selector": "greedy"
    },
    "firefly_dynamic": {
        "parameters": {
            "distribution": "gaussian",
            "alpha": 0.5,
            "beta": 1.5,
            "gamma": 50.0
        },
        "selector": "probabilistic"
    },
    "genetic_crossover": {
        "parameters": {
            "pairing": "cost",
            "crossover": "uniform",
            "mating_pool_factor": 0.3
        },
        "selector": "all"
    },
    "genetic_mutation": {
        "parameters": {
            "scale": 1.5,
            "elite_rate": 0.2,
            "mutation_rate": 0.3,
            "distribution": "gaussian"
        },
        "selector": "metropolis"
    }
}
```