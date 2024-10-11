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

class NoCodeException(Exception):
    pass
 
class RagCustumhys:
    """
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """
    def __init__(
        self,
        f,
        role_prompt="",
        task_prompt="",
        experiment_name="",
        elitism=True,
        feedback_prompt="",
        budget=100,
        client ="",
        model="deepseek-coder-v2",
        log=True,
    ):
        """
        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            feedback_prompt (str): Prompt to guide the model on how to provide feedback on the generated algorithms.
            budget (int): The number of generations to run the evolutionary algorithm.
            model (str): The model identifier from OpenAI or ollama to be used.
            log (bool): Flag to switch of the logging of experiments.
        """
        #self.client = LLMmanager(model)
        self.model = model
        self.f = f  # evaluation function, provides a string as feedback, a numerical value (higher is better), and a possible error string.
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve the benchmark problems provided by the user."
        if task_prompt == "":
            self.task_prompt = """
You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the {fun} optimization problem using only the operators and selectors from the parameters_to_take.txt file.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.


INSTRUCTIONS:
1. Use only the function: bf.{experiment_name}
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


fun = bf.{experiment_name}
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

# Short explanation and justification:
# [Your explanation here, each line starting with '#']
```
"""

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent program.
        """
        self.last_error = ""
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.task_prompt},
        ]

        try:
            solution, name, algorithm_name_long = self.llm(session_messages)
            self.last_solution = solution
            (
                self.last_feedback,
                self.last_fitness,
                self.last_error,
            ) = self.evaluate_fitness(solution, name, algorithm_name_long)
        except NoCodeException:
            self.last_fitness = -np.Inf
            self.last_feedback = "No code was extracted."
        except Exception as e:
            self.last_fitness = -np.Inf
            self.last_error = repr(e) + traceback.format_exc()
            self.last_feedback = f"An exception occured: {self.last_error}."
            print(self.last_error)
        self.generation += 1

        self.best_solution = self.last_solution
        self.best_fitness = self.last_fitness
        self.best_error = self.last_error
        self.best_feedback = self.last_feedback

    def llm(self, session_messages):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, and its full descriptive name.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if self.log:
            self.logger.log_conversation(
                "RAG-CUSTUMHYS", "\n".join([d["content"] for d in session_messages])
            )

        message = self.client.chat(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)
        new_algorithm = self.extract_algorithm_code(message)

        algorithm_name = re.findall("class\\s*(\\w*)\\:", new_algorithm, re.IGNORECASE)[
            0
        ]
        algorithm_name_long = self.extract_algorithm_name(message)
        if algorithm_name_long == "":
            algorithm_name_long = algorithm_name
        # todo rename algorithm
        self.last_solution = message
        # extract algorithm name and algorithm
        return new_algorithm, algorithm_name, algorithm_name_long

    def evaluate_fitness(self, solution, name, long_name):
        """
        Evaluates the fitness of the provided solution by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            solution (str): The solution code to evaluate.
            name (str): The name of the algorithm.
            long_name (str): The full descriptive name of the algorithm.

        Returns:
            tuple: A tuple containing feedback (string), fitness (float), and error message (string).
        """
        # Implement fitness evaluation and error handling logic.
        if self.log:
            self.logger.log_code(self.generation, name, solution)
        feedback, fitness, error = self.f(solution, name, long_name, self.logger)
        self.history += f"\nYou already tried {long_name}, with score: {fitness}"
        if error != "":
            self.history += f" with error: {error}"
        return feedback, fitness, error

    def construct_prompt(self):
        """
        Constructs a new session prompt for the language model based on the best or the latest solution,
        depending on whether elitism is enabled.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        if self.elitism:
            solution = f"The best so far algorithm is as follows: \n```\n{self.best_solution}\n```\n"
            feedback = self.best_feedback
        else:
            solution = f"The last tried algorithm is as follows: \n```\n{self.last_solution}\n```\n"
            feedback = self.last_feedback

        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.task_prompt},
            {"role": "user", "content": self.history},
            {"role": "assistant", "content": solution},
            {"role": "user", "content": feedback},
            {"role": "user", "content": self.feedback_prompt},
        ]
        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

    def update_best(self):
        """
        Updates the record of the best solution found so far if the latest solution has a higher fitness.
        This method checks and compares the fitness of the latest solution against the best-known fitness.
        """
        if self.best_fitness <= self.last_fitness or self.last_fitness == -np.Inf:
            self.best_solution = self.last_solution
            self.best_fitness = self.last_fitness
            self.best_error = self.last_error

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            print(message, "contained no ``` code block")
            raise NoCodeException

    def extract_algorithm_name(self, message):
        """
        Extracts algorithm name from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Name:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""

    def run(self):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        self.initialize()
        while self.generation < self.budget:
            new_prompt = self.construct_prompt()
            try:
                self.last_solution, name, algorithm_name_long = self.llm(new_prompt)
                (
                    self.last_feedback,
                    self.last_fitness,
                    self.last_error,
                ) = self.evaluate_fitness(self.last_solution, name, algorithm_name_long)
            except NoCodeException:
                self.last_fitness = -np.Inf
                self.last_feedback = "No code was extracted."
                self.last_error = (
                    "The code should be encapsulated with ``` in your response."
                )
            except Exception as e:
                self.last_fitness = -np.Inf
                self.last_error = repr(e)
                self.last_feedback = f"An exception occured: {self.last_error}."

            self.update_best()
            self.generation = self.generation + 1

        return self.best_solution, self.best_fitness
