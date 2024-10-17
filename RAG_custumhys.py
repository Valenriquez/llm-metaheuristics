import re
import traceback
import numpy as np
import ollama
import chromadb
import os
import benchmark_func as bf
import sys
import datetime
import subprocess
import time
import logging
import llm
from ollama import ResponseError

class NoCodeException(Exception):
    pass
 
 
class MetaheuristicGenerator:

    def __init__(self, experiment_name, model="deepseek-coder-v2", max_iterations=7):
        #self.experiment_name = function
        self.model = model
        self.max_iterations = max_iterations
        self.client = chromadb.Client()
        self.python_files_collection = self.client.create_collection(name="algorithm_creation")
        self.feedback_collection = self.client.create_collection(name="feedback_collection")
        self.experiment_name = experiment_name

        
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        #sys.path.append('llm-metaheuristics/algorithm_creation')
        
        #self.fun = getattr(bf, function)(2)
        
        self.python_files_dir = 'llm-metaheuristics/algorithm_creation'
        self.process_python_files()
    
    def process_python_files(self):
        for filename in os.listdir(self.python_files_dir):
            if filename.endswith('.py') or filename.endswith('.txt'):
                file_path = os.path.join(self.python_files_dir, filename)
                file_content = self.read_python_file(file_path)
                
                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
                embedding = response.get("embedding")
                
                if embedding:
                    self.python_files_collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": filename}]
                    )
                else:
                    print(f"Warning: Empty embedding generated for {filename}")

    def read_python_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
        #self.client = LLMmanager(model)
        #self.model = model
        #self.f = f  # evaluation function, provides a string as feedback, a numerical value (higher is better), and a possible error string.
    def generate_prompt(self):
        self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve the benchmark problems provided by the user."
        self.task_prompt = """
            You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the {fun} optimization problem using only the operators and selectors from the parameters_to_take.txt file.
            IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
            DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.


            INSTRUCTIONS:
            1. Use only the function: bf.{self.experiment_name}
            2. Use only operators and selectors from parameters_to_take.txt. 
            3. Use only the parameters of the operator chosen from parameters_to_take.txt. 
            4. The options inside the array are the ones you can choose from to fill each parameter.
            5. Only use one variable per parameter
            6. Do Not use the whole array when writing the variable of the parameter.
            7. Write the variables without an array format
            8. Write the variable as a float or string format.
            9. The search space is between -1.0 (lower bound) and 1.0 (upper bound)
            10. Set num_iterations to 100
            12. Each operator must have its own selector
            13. Fill all parameters for the chosen operator with your best recommendations. You must read the complete parameters_to_take.txt file to know all the parameters for each operator.
            14. You can use Two operator per metaheuristic if you think that is the best option, but do not use more than three operators.
            15. Create only one metaheuristic per response
            16. DO NOT use any information or knowledge outside of what is provided in the parameters_to_take.txt file

            FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

            ```
            # Name: [Your chosen name for the metaheuristic]
            # Code:

            import sys
            sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
            import benchmark_func as bf
            import metaheuristic as mh


            fun = bf.{self.experiment_name}
            prob = fun.get_formatted_problem()

            heur = [
                        ( # Search operator 1
                        '[operator_name]',
                        {
                            'parameter1': value1,
                            'parameter2': value2,
                            ... more parameters as needed
                        },
                        '[selector_name]'
                        ),
                        (  
                        '[operator_name]',
                        {
                            'parameter1': value1,
                            'parameter2': value2,
                            ... more parameters as needed
                        },
                        '[selector_name]'
                    )
                ]

            met = mh.Metaheuristic(prob, heur, num_iterations=100)
            met.verbose = True
            met.run()
            print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))
            ```

            # Short e xplanation and justification:
            # [Your explanation here, each line starting with '#']
            ```
            """
        # Combine role_prompt and task_prompt
        full_prompt = f"{self.role_prompt}\n\n{self.task_prompt}"
        return full_prompt
            
    def self_refine(self, initial_prompt, data, output_folder, max_iterations=7):
    # Initialize a collection for storing feedback
        
        #feedback_collection = chromadb.Client().create_collection(name="feedback_collection")
        
        # Initialize a collection for Python files if not already done
        
        # python_files_collection = chromadb.Client().get_or_create_collection(name="algorithm_creation")
        
        try:
            current_output = ollama.generate(
                model="deepseek-coder-v2",
                prompt=f"Using this data: {data}. Respond to this prompt: {initial_prompt}"
            )
        except ResponseError as e:
            self.logger.error(f"Ollama generate error: {str(e)}")
            return f"Error generating output: {str(e)}"
        
        print(current_output['response'])
        
        # Write initial output  # commenting to avoid to much files 
        #write_output_to_file(current_output['response'], output_folder, 0)
        
        for i in range(max_iterations):
            execution_result = self.execute_generated_code(current_output['response'], output_folder, i)
            
            # Add the current output and execution result to the feedback collection
            feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
            self.feedback_collection.add(
                ids=[f"iteration_{i}"],
                embeddings=[feedback_embedding['embedding']],
                documents=[current_output['response'] + "\n" + execution_result],
                metadatas=[{"iteration": i}]
            )
            
            # Retrieve relevant feedback from previous iterations
            query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
            
            # Ensure n_results is at least 1
            n_results = max(1, min(i, 7))
            relevant_feedback = self.feedback_collection.query(
                query_embeddings=[query_embedding['embedding']],
                n_results=n_results
            )
            
            # Retrieve all Python files
            if self.python_files_collection.count() > 0:
                total_docs = self.python_files_collection.count()
                relevant_files = self.python_files_collection.query(
                    query_embeddings=[query_embedding['embedding']],
                    n_results=total_docs  # Retrieve all documents
                )
                
                # Sort the results by relevance score (if available)
                if 'distances' in relevant_files:
                    sorted_indices = sorted(range(len(relevant_files['distances'][0])), 
                                            key=lambda k: relevant_files['distances'][0][k])
                    
                    sorted_documents = [relevant_files['documents'][0][i] for i in sorted_indices]
                    relevant_files['documents'] = [sorted_documents]
                
                # Limit the number of documents to include in the prompt if necessary
                max_docs_to_include = 2  # Adjust this number as needed
                relevant_files['documents'][0] = relevant_files['documents'][0][:max_docs_to_include]
            else:
                relevant_files = {"documents": ["No relevant Python files found."]}
            
            # Construct the refinement prompt with relevant feedback and Python files
            refinement_prompt = f"""
            IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
            DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
            You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

            {current_output['response']}
            The code was executed with the following result:
            {execution_result}
            You must fix the results. I need the metaheuristic to run correctly. 
            Here is relevant feedback from previous iterations:
            {relevant_feedback['documents']}

            Here are relevant Python files that might be helpful:
            {relevant_files['documents']}

            Please analyze this output and suggest improvements and corrections. 
            Please DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ``` in the response.
            Please DO NOT USE ANY operators or parameters that are not in the parameters_to_take.txt file.
            This is the parameters_to_take.txt file:
            {data}
            IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
            Use the same template as the one provided before, which is:
            
            # Name: [Your chosen name for the metaheuristic]
            # Code:

            import sys
            sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
            import benchmark_func as bf
            import metaheuristic as mh


            fun = bf.Rastrigin(2)
            prob = fun.get_formatted_problem()

            heur = [
                ( # Search operator 1
                '[operator_name]',
                {{ 
                    'parameter1': value1,
                    'parameter2': value2,
                    ... more parameters as needed
                }},
                '[selector_name]'
                ),
                (  
                '[operator_name]',
                {{
                    'parameter1': value1,
                    'parameter2': value2,
                    ... more parameters as needed
                }},
                '[selector_name]'
            )
        ]

            met = mh.Metaheuristic(prob, heur, num_iterations=100)
            met.verbose = True
            met.run()
            print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))


            # Short explanation and justification:
            # [Your explanation here, each line starting with '#']

            REMEMBER: 
            1. EVERY EXPLANATION MUST START WITH '#'. 
            2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```.
            3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
            DO NOT INVENT ANY NEW INFORMATION.
            4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
            5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
            6. If you ever use genetic crossover, you must use genetic mutation as well. 
            7. Verifying that only operators and parameters from parameters_to_take.txt are used.
            8. Checking for any logical errors or inconsistencies.
            9. Improving the explanation and justification.

            Provide your refined version of the entire output, not just the changes.
            """

            try:
                refined_output = ollama.generate(
                    model="deepseek-coder-v2",
                    prompt=refinement_prompt
                )
            except ResponseError as e:
                self.logger.error(f"Ollama generate error: {str(e)}")
                return f"Error generating refined output: {str(e)}"
            
            # Write refined output
            self.write_output_to_file(refined_output['response'], output_folder, i+1)
            
            # Check if the refinement made significant changes
            if refined_output['response'].strip() == current_output['response'].strip():
                print(f"No significant changes after iteration {i+1}. Stopping refinement.")
                break
            
            current_output = refined_output
            print(f"Completed refinement iteration {i+1}")
        
        return current_output['response']
    
    def write_output_to_file(self, output, folder, iteration):
        file_name = f'ollama_output_iteration_{iteration}.py'
        file_path = os.path.join(folder, file_name)
        try:
            #with open(file_path, 'w') as file:
            #     file.write(output)
            print(f"Output for iteration {iteration} has been written to {file_path}")
        except Exception as e:
            print(f"An error occurred while writing iteration {iteration} to file: {e}")

    def execute_generated_code(self, code, output_folder, iteration):
        file_name = f'execution_iteration_{iteration}.py'
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(['python', file_path], capture_output=True, text=True, timeout=30)
            execution_result = f"Exit code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            
            # Write execution result to a separate file
            result_file_name = f'execution_result_{iteration}.txt'
            result_file_path = os.path.join(output_folder, result_file_name)
            with open(result_file_path, 'w') as f:
                f.write(execution_result)
            
            return execution_result
        except subprocess.TimeoutExpired:
            return "Execution timed out after 30 seconds"
        except Exception as e:
            return f"An error occurred during execution: {str(e)}"
        

    def construct_refinement_prompt(self, current_output, execution_result, data, feedback_embedding, iteration):
        f"""
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

        {current_output['response']}
        The code was executed with the following result:
        {execution_result}
        You must fix the results. I need the metaheuristic to run correctly. 
        Here is relevant feedback from previous iterations:
        {feedback_embedding['documents']}

        Here are relevant Python files that might be helpful:
        {data['documents']}

        Please analyze this output and suggest improvements and corrections. 
        Please DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ``` in the response.
        Please DO NOT USE ANY operators or parameters that are not in the parameters_to_take.txt file.
        This is the parameters_to_take.txt file:
        {data}
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
        Use the same template as the one provided before, which is:
        
        # Name: [Your chosen name for the metaheuristic]
        # Code:

        import sys
        sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
        import benchmark_func as bf
        import metaheuristic as mh


        fun = bf.Rastrigin(2)
        prob = fun.get_formatted_problem()

        heur = [
            ( # Search operator 1
            '[operator_name]',
            {{ 
                'parameter1': value1,
                'parameter2': value2,
                 ... more parameters as needed
            }},
            '[selector_name]'
            ),
            (  
            '[operator_name]',
            {{
                'parameter1': value1,
                'parameter2': value2,
                 ... more parameters as needed
            }},
            '[selector_name]'
        )
      ]

        met = mh.Metaheuristic(prob, heur, num_iterations=100)
        met.verbose = True
        met.run()
        print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))


        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']

        REMEMBER: 
        1. EVERY EXPLANATION MUST START WITH '#'. 
        2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```.
        3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
        DO NOT INVENT ANY NEW INFORMATION.
        4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
        5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
        6. If you ever use genetic crossover, you must use genetic mutation as well. 
        7. Verifying that only operators and parameters from parameters_to_take.txt are used.
        8. Checking for any logical errors or inconsistencies.
        9. Improving the explanation and justification.

        Provide your refined version of the entire output, not just the changes.
        """

    def write_output_to_file(self, output, folder, iteration):
        file_name = f'ollama_output_iteration_{iteration}.py'
        file_path = os.path.join(folder, file_name)
        try:
            with open(file_path, 'w') as file:
                file.write(output)
            print(f"Output for iteration {iteration} has been written to {file_path}")
        except Exception as e:
            print(f"An error occurred while writing iteration {iteration} to file: {e}")

    def execute_generated_code(self, code, output_folder, iteration):
        file_name = f'execution_iteration_{iteration}.py'
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(['python', file_path], capture_output=True, text=True, timeout=30)
            execution_result = f"Exit code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            
            result_file_name = f'execution_result_{iteration}.txt'
            result_file_path = os.path.join(output_folder, result_file_name)
            with open(result_file_path, 'w') as f:
                f.write(execution_result)
            
            return execution_result
        except subprocess.TimeoutExpired:
            return "Execution timed out after 30 seconds"
        except Exception as e:
            return f"An error occurred during execution: {str(e)}"
        
    def run(self):
        self.logger.debug("Starting main execution")
        try:
            prompt = self.generate_prompt()
            self.logger.debug(f"Generated prompt: {prompt}")  # Log the generated prompt


            response = ollama.embeddings(
                prompt=prompt,
                model="mxbai-embed-large"
            )
                # Check if the embedding is empty
            if not response.get("embedding"):
                self.logger.error("Generated embedding is empty")
        
            
            results = self.python_files_collection  .query(
                query_embeddings=[response["embedding"]],
                n_results=1
            )
            
            # Check if results are empty
            if not results['documents']:
                self.logger.error("No results found in the collection")
            
            data = results['documents'][0][0]


            current_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')

            os.makedirs(output_folder)
            self.logger.info(f"Created new folder: {output_folder}")

        
            try:
                refined_output = self.self_refine(prompt, data, output_folder, max_iterations=7)
                self.logger.info("Final refined output:")
                print(refined_output)
            except Exception as e:
                self.logger.error(f"An error occurred during self-refinement: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            raise

        self.logger.debug("Main execution completed")
            

if __name__ == "__main__":
    generator = MetaheuristicGenerator("Rastrigin")
    generator.run()






                #    self.feedback_prompt = (
    #       f"Either refine or redesign to improve the solution (and give it a distinct name). Give the response in the format:\n"
    #        f"# Name: <name>\n"
    #        f"# Code: <code>"
    #    )
        #self.budget = budget
    #    self.generation = 0
    #    self.best_solution = None
    #    self.best_fitness = -np.Inf
    #    self.best_error = ""
    #    self.last_error = ""
    #    self.last_solution = ""
    #    self.history = ""
        
    #    def rag(self, session_messages):
    #        # Your existing code to generate the output
    #        response = ollama.embeddings(
    #            prompt=prompt,
    #            model="mxbai-embed-large"
    #        )
    #        results = collection.query(
    #            query_embeddings=[response["embedding"]],
    #            n_results=1
    #        )
    #        data = results['documents'][0][0]
    #        return self.client.chat(session_messages)
        
