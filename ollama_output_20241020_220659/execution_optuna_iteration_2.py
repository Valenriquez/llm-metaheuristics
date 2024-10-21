**Objetivo:** Optimizar los hiperparámetros de un algoritmo genético para minimizar la función de Rastrigin.

**Metodología:**

* Se utiliza el algoritmo de optimización Optuna para encontrar los mejores hiperparámetros.
* Se define una función objetivo que evalúa el rendimiento del algoritmo genético con diferentes valores de hiperparámetros.
* Se realiza un estudio de optimización con 50 pruebas.

**Hiperparámetros:**

* `differential_mutation`: expresión de mutación diferencial (rand, best, current, etc.).
* `num_rands`: Número de soluciones aleatorias en la mutación diferencial.
* `factor`: Factor de escala para las soluciones mutadas.
* `pairing`: Pareamiento de poblaciones (selected_pairing).
* `crossover`: Tipo de cruza (selected_crossover).
* `mating_pool_factor`: Factor de tamaño de la piscina de selección.
* `factor`: Factor de dinámica de la colmena.
* `self_conf`: Confiabilidad individual.
* `swarm_conf`: Confiabilidad de la colmena.
* `version`: Versión de la colmena (selected_version).
* `distribution`: Distribución de probabilidad (selected_distribution).

**Algoritmo genético:**

* Se utiliza un algoritmo genético con una población inicial de 50 soluciones.
* Se realiza una selección de la población, un cruce y una mutación.
* Se realiza un seguimiento del rendimiento de la población durante 100 iteraciones.

**Resultados:**

* Los mejores hiperparámetros encontrados son:
```
{'differential_mutation': 'rand', 'num_rands': 1, 'factor': 1.0, 'pairing': 'selected_pairing', 'crossover': 'selected_crossover', 'mating_pool_factor': 0.5, 'factor': 0.6, 'self_conf': 2.0, 'swarm_conf': 2.0, 'version': 'selected_version', 'distribution': 'selected_distribution'}
```
* El mejor rendimiento encontrado es:
```
-20.44444444444444
```

**Conclusión:**

Los mejores hiperparámetros encontrados optimizan el rendimiento del algoritmo genético para minimizar la función de Rastrigin.