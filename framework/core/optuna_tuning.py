import optuna
import os
import ollama
import re

"""

"""

class OptunaTuner:
    def __init__(self, output_folder, number_iteration, problem_id, dimensions, num_of_agents, model, model_embed, optuna_collection):
        self.number_iteration = number_iteration
        self.output_folder = output_folder
        self.problem_id = problem_id
        self.dimensions = dimensions
        self.num_of_agents = num_of_agents
        self.model = model
        self.model_embed = model_embed
        self.optuna_collection = optuna_collection

        # Read nce during initialization
        input_file_path = os.path.join(self.output_folder, f'execution_iteration_{self.number_iteration}.py')
        with open(input_file_path, 'r') as f:
            self.file_contents = f.read()

    def extract_code_from_code_with_optuna(self, file_contents):
        pattern = r'heur\s*=\s*\[(.*?)\]' 
        match = re.search(pattern, file_contents, re.DOTALL)
        if match:
            extracted_content = match.group(1).strip()  
            print("MATCH", extracted_content)

            output = ollama.generate(
            model = self.model,
            prompt = f"""Modify this metaheuristic: {extracted_content}, DO NOT ADD any new parameters. 
                            You must make the following changes based on the parameter type:

                        1. **If the parameter provides a numeric value:**
                        - Change it to the format:
                            name_variable = trial.suggest_float('name_variable', lower_limit, upper_limit)
                        - Use appropriate ranges for `lower_limit` and `upper_limit`. 
                            - **Rules for ranges:**
                            - For **radius**, the maximum is 0.9.
                            - For **angle**, the maximum is 25.
                            - For **swarm_conf** or **self_conf**, the maximum is 3.
                        - **Incorrect Example:**
                            'name_variable': trial.suggest_categorical('name_variable', ['2.54'])
                        - **Correct Example:**
                            'beta': trial.suggest_float('beta', 1.5, 4.5),
                        - Always include a comma after the modified parameter.

                        2. **If the parameter provides a category:**
                        - Modify it to the format:
                            'category_name': trial.suggest_categorical('category_name', ['option_1', 'option_2', 'option_3'])
                        - Include at least **three options**. ALWAYS USE MORE THAN ONE option.
                        - **Incorrect Example:**
                            'category_name': trial.suggest_categorical('category_name', ['option_1'])
                        - **Correct Example:**
                            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian','levy'])


                        3. **Rules for specific category types:**
                        - "version": "inertial", "constriction"
                        - "distribution": "uniform", "gaussian", "levy"
                        - "pairing": "rank", "cost", "random", "tournament_2_100"
                        - "crossover": "single", "two", "uniform", "blend", "linear_0.5_0.5"
                        - "expression": "rand", "best", "current", "current-to-best", "rand-to-best", "rand-to-best-and-current"

                        4. **General Guidelines:**
                        - Do not change anything else in the metaheuristic. Only modify the parameters as instructed.
                        - Do not add extra words, explanations, or paragraphs. Follow these instructions strictly.
                        """
            ) 
            # Deleted from the prompt: , take a look to the other parameters provided for the operators and selectors: {optuna_data}
            #print("is it doing it wrong", output['response'])
            return output['response']
        else:
            return None

    def execute_generated_code(self, code, output_folder, number_iteration, is_optuna):
        # Placeholder for your actual execution logic
        print(f"Executing generated code (iteration {number_iteration}, optuna={is_optuna})...")
        exec(code, globals())

    def generation(self):
        # Step 1: Get optuna template via embedding search
        optuna_template_response = ollama.embeddings(
            prompt="give me the optuna template",
            model=self.model_embed
        )

        results = self.optuna_collection.query(
            query_embeddings=[optuna_template_response["embedding"]],
            n_results=1
        )
        optuna_template = results['documents'][0][0]

        # Step 2: Extract heuristics from current file
        extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)

        # Step 3: Build prompt dynamically
        optuna_task_prompt = f"""
You are an expert in natural computing. Your task is to generate code adhering strictly to the given template. 

### Rules:
1. **Plain Text Only**: Do not include Markdown or triple backticks.
2. **No Deviations**: Follow the provided template exactly—no additions, modifications, or extra explanations.
3. **Use the following template**: {optuna_template}
### Template Modifications:
- Replace with:
    problem_id = {self.problem_id}    
    instance = 1
    dimension = {self.dimensions}     
    num_agents = {self.num_of_agents}    
    num_iterations = 100
    num_replicas = 10
- Replace `def objective(trial):` with:
    def objective(trial):
        heur = [
            {extracted_metaheuristic}
        ]
- Use this format for `heur`:
    heur = [
        ('[operator_name]', {{'parameter1': value1}}, '[selector_name]')
    ]
"""

        full_prompt = optuna_task_prompt + f"""
Remember to put:
    performance = evaluate_sequence_performance(
        heur,
        prob,
        num_agents={self.num_of_agents},
        num_iterations=100,
        num_replicas=30
    )
"""

        # Step 4: Generate code using LLM
        output = ollama.generate(model=self.model, prompt=full_prompt)
        response = output.get("response", "").strip()

        # Step 5: Verify and execute
        checker_variable = 0
        if response:
            checker_variable += 1
            print("checker_variable--OPTUNA--->>>>>", checker_variable)
            self.execute_generated_code(response, self.output_folder, self.number_iteration, is_optuna=True)

        # You may handle the checker loop outside this method
        return checker_variable

    def run(self, n_trials=50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params
