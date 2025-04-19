import logging
import math
import datetime
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

from feedback import Feedback
from collection_manager import CollectionManager
from execute_code import CodeExecutor, ExecutionConfig  
from exploration import Exploration
from exploitation import Exploitation

class MainFramework:
    def __init__(
        self,
        problem_id: int,
        dimensions: int,
        number_iteration: int,
        model: str = "qwen2.5-coder:latest",
        model_embed: str = "all-minilm:latest",
        total_budget: int = 100,
        break_tries: int = 8,
        base_path: str = "."  # default current directory
    ):
        self.problem_id = problem_id
        self.dimensions = dimensions
        self.number_iteration = number_iteration
        self.model = model
        self.model_embed = model_embed

        # Metaheuristic and loss tracking
        self.hyperparameters = ""
        self.made_metaheuristic = ""
        self.the_bool = True
        self.old_losses = []
        self.new_losses = []
        self.history_append = []

        # Output handling
        self.folder_name: Path = Path()
        self.extracted_metaheuristic = ""
        self.file_contents = ""

        # Execution feedback
        self.file_result = 1
        self.file_result_error = ""

        # Tuning and agents
        self.total_budget = total_budget
        self.break_tries = break_tries
        self.num_of_agents = 10 * self.dimensions + math.floor(20 * math.log(self.dimensions + 1))

        # Logging setup
        self.logger = logging.getLogger(__name__)

        # === Base Path and Collection Manager ===
        self.base_path = Path(base_path)
        self.collection_manager = CollectionManager()
        

        # === Defer Executor Initialization Until Folder Exists ===
        self.executor = None

        # === Defer Exploration Until Output Path is Ready ===
        self.exploration_instance = None

    def create_output_folder(self) -> Path:
        current_dir = Path(__file__).parent.resolve()
        output_folder_parent = current_dir / 'results'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = output_folder_parent / f'ollama_output_{self.problem_id}_{self.dimensions}_{timestamp}'
        output_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created new folder: {output_folder}")
        return output_folder

    def run(self):
        self.logger.debug("Starting main execution")
        try:
            self.folder_name = self.create_output_folder()

            chroma_client = chromadb.Client()
            google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyClnhvj-6aQdDS2qcheEoep2SiCUXvQz-I")
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
                        
            self.operators_collection = chroma_client.create_collection(name="operators_collection", embedding_function=google_ef)
            self.metaheuristic_template_collection = chroma_client.create_collection(name="metaheuristic_template_collection", embedding_function=google_ef)
            self.feedback_collection = chroma_client.create_collection(name="feedback_collection", embedding_function=sentence_transformer_ef)
            self.optuna_collection = chroma_client.create_collection(name="ioh_optuna_builder", embedding_function=google_ef)

            self.collection_manager.function_create_collection("operatorsData", self.operators_collection, "all-minilm:latest")
 
    
            execution_config = ExecutionConfig(output_dir=str(self.folder_name))
            self.executor = CodeExecutor(execution_config)

            self.exploration_instance = Exploration(
                problem_id=self.problem_id,
                dimensions=self.dimensions,
                num_of_agents= self.num_of_agents, 
                feedback_collection = self.feedback_collection,
                operators_collection = self.operators_collection,
                metaheuristic_template_collection = self.metaheuristic_template_collection
            )

            self.exploitation_instance = Exploitation(
                problem_id=self.problem_id,
                dimensions=self.dimensions,
            )
 

            for i in range(self.number_iteration):
                self.logger.debug(f"Starting exploration iteration {i}")

                generated_code = self.exploration_instance.exploration(i)

                result = self.executor.execute_generated_code(
                    code=generated_code,
                    number_iteration=i
                )
                self.logger.info(f"Iteration EXPLORATION {i} result:\n{result}")

                enhanced_code = self.exploitation_instance.exploitation(i)


                feedback_manager = Feedback(
                    folder_feedback=self.folder_name,
                    collection=self.feedback_collection
                )

                

        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise


if __name__ == "__main__":
    generator = MainFramework(9, 3, 3)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)