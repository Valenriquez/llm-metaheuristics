import logging
from pathlib import Path
import math
import pathlib
import datetime

class MainFramework:
    def __init__(self, problem_id, dimensions, max_iterations, model="qwen2.5-coder:latest", model_embed="all-minilm:latest"):
        self.model = model
        self.model_embed = model_embed
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.problem_id = problem_id
        self.dimensions = dimensions
        self.hyperparameters = ""
        self.best_performance = 500
        self.performance_found = 200
        self.made_metaheuristic = ""
        self.the_bool = True

        self.old_losses = []
        self.new_losses = []
        self.history_append = []

        ## OUTPUT INFO
        self.folder_name = ""
        self.extracted_metaheuristic = ""

        ## LOGGER
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        ## ERROR CHECKER
        self.file_result = 1
        self.file_result_error = ""
        self.total_budget = 100 # (?)

        ## NEW APPROACH
        self.break_tries = 8 # for Optuna Enhancement only
        self.num_of_agents = 10 * self.dimensions + math.floor(20 * math.log(self.dimensions + 1))
        self.standard_dev = 0

        ## FOR OPTUNA
        self.file_contents = "" 

    def run(self):
        self.logger.debug("Starting main execution")
        try:
            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = pathlib.Path(__file__).parent.resolve()
            output_folder_parent = current_dir / 'results'

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = output_folder_parent / f'ollama_output_{self.problem_id}_{self.dimensions}_{timestamp}'
            self.folder_name = output_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                self.logger.debug(f"Starting exploration iteration {i}")
                self.exploration(output_folder, i)
            # -------------------- 
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise         
