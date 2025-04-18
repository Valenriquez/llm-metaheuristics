from dataclasses import dataclass
import json

@dataclass
class Feedback:
    collection: any   

    def store_feedback(self, iteration: int, metaheuristic_code: str, hyperparams: dict, performance: float):
        expected_keys = {
            'scale', 'elite_rate', 'mutation_rate', 'probability',
            'gravity', 'alpha', 'beta', 'dt', 'mating_pool_factor',
            'num_rands', 'pairing', 'crossover', 'radius', 'angle',
            'sigma', 'factor', 'self_conf', 'swarm_conf', 'version',
            'expression', 'distribution'
        }

        filtered_hyperparams = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in hyperparams.items()
            if k in expected_keys and v != ''
        }

        hparams_str = json.dumps(filtered_hyperparams, indent=4)
        perf_str = str(performance)

        self.collection.add(
            documents=[
                f"Iteration: {iteration}, Parameters:\n{hparams_str}\nPerformance: {performance}",
                metaheuristic_code,
                perf_str
            ],
            metadatas=[{"hnsw:space": "cosine"}] * 3,
            ids=[
                f"id_parameters_{iteration}",
                f"id_metaheuristic_file_{iteration}",
                f"id_performance_found_{iteration}"
            ]
        )
        print(f"✅ Feedback stored for iteration {iteration}")
