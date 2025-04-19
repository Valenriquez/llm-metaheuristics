from dataclasses import dataclass
import json
import re

@dataclass
class Feedback:
    folder_feedback: str
    collection: any 

    def get_hyperparameters(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code_file = file.read()
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None, None

        # Regex patterns
        hyperparameters_pattern = r"Mejores hiperparámetros encontrados:\n({.*?})"
 
        hyperparameters_match = re.search(hyperparameters_pattern, code_file, re.DOTALL)
        hyperparameters_dict = eval(hyperparameters_match.group(1)) if hyperparameters_match else None

        return hyperparameters_dict

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
        print(f"Feedback stored for iteration {iteration}")
