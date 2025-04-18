```python
{
  "operators": [
    {
      "name": "random_search",
      "parameters": {
        "scale": 0.1,
        "distribution": "gaussian"
      },
      "selector": "probabilistic"
    },
    {
      "name": "central_force_dynamic",
      "parameters": {
        "gravity": 0.01,
        "alpha": 0.1,
        "beta": 2.0,
        "dt": 0.5
      },
      "selector": "metropolis"
    },
    {
      "name": "differential_mutation",
      "parameters": {
        "expression": "rand-to-best-and-current",
        "num_rands": 3,
        "factor": 1.2
      },
      "selector": "greedy"
    }
  ]
}
```