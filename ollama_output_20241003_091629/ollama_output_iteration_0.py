 # Name: Adaptive Random Search (ARS)
# Code:
```python
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1 - Gravitational Search Algorithm (GSA)
            'gravitational_search',
            {
                'gravity': 0.5,
                'alpha': 0.02
            },
            'probabilistic'
            ),
            (  
            'random_flight',
            {
                'scale': 0.8,
                'distribution': 'levy',
                'beta': 1.5
            },
            'metropolis'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
```
# Short explanation and justification:
The Adaptive Random Search (ARS) metaheuristic is designed to explore the search space using a combination of two main operators: Gravitational Search Algorithm (GSA) and Random Flight. 

1. **Gravitational Search Algorithm (GSA):** This operator simulates the gravitational force between masses, where 'gravity' controls the strength of this force, influencing the movement of particles towards better solutions. The parameter 'alpha' is set to 0.02 to adjust the acceleration due to gravity. GSA uses a probabilistic selector ('probabilistic') which allows it to act randomly in its search but with a bias influenced by gravitational forces.

2. **Random Flight:** This operator mimics random walks and is controlled by 'scale', which defines the amplitude of these random steps, and 'beta' which affects the distribution of these steps. The distribution type here is set to 'levy', which introduces long jumps in search space that can help avoid local minima. The selector used here is 'metropolis', which allows for a probabilistic acceptance criterion based on energy differences in the search space.

Both operators are chosen because they balance exploration and exploitation: GSA ensures that particles move towards better solutions with a controlled randomness, while Random Flight introduces global exploration by means of non-uniform step sizes influenced by levy distribution flights. This combination helps the metaheuristic to effectively navigate complex landscapes, leveraging both gravitational attraction towards promising areas and random jumps across the search space.

The total number of iterations is set to 200, allowing for a thorough exploration while also ensuring computational efficiency without unnecessary runtime. The use of probabilistic selectors further enhances the adaptability and robustness of the algorithm by adjusting the balance between exploitation and exploration based on the current state of the search.