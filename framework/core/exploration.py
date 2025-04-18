import ollama 
import os
from dataclasses import dataclass
from chromadb.api.models.Collection import Collection

"""
Exploration:  Will create a metaheuristic and search for new creations. After the first iteration it will look on the feedback for inspiration on better 
metaheuirsics creation. 
- Will have access to feedback
- Will get hyperparameters for the refinement 
"""

@dataclass
class Exploration:
    problem_id: int
    dimensions: int
    model: str = "qwen2.5-coder:latest"
    model_embed: str = "all-minilm:latest"
    feedback_collection:  Collection = None 
    operators_collection:  Collection = None 
    metaheuristic_template_collection:  Collection = None 

    def exploration(self, number_iteration):
        print("Beginning with exploration number --->",  number_iteration)
        output_response = None

        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        output_operators = ollama.embeddings(
        prompt=f"Remember to write the operators' names",
        model=self.model_embed
        )
        results = self.operators_collection.query(
        query_embeddings=[output_operators["embedding"]],
        n_results=1
        )
        data  = results['documents'][0][0]
        print("operators data:  ", data)
        
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
     
          # there won´t be any feedback else:
      
        # Query for the Feedback - - - - - - - - (asks for the metaheuristics and feedback) - - - - - - - - - - -
        output_feedback = ollama.embeddings(
        prompt="Give me the best operators for the given metaheuristic",
        model=self.model_embed
        )
        results = self.operators_collection.query(
        query_embeddings=[output_feedback["embedding"]],
        n_results=3
        )
        answer = results['documents'][0][0]
        print("The best operators for the given problem---", answer)
        # Query for the Feedback - - - - - - - - - - - - - - - - - - -
        output = ollama.generate(
                model=self.model,
                prompt = f"""
                ### INSTRUCTIONS FOR METAHEURISTIC REFINEMENT:
                1. Analyze the provided metaheuristic data and performance constraints to create a refined metaheuristic.
                2. Focus on strategies that either **improve performance**, **reduce computational cost**, or achieve a balance between the two.
                3. Suggest specific modifications to:
                    - Operators
                    - Parameters
                    - Selectors
                Base your suggestions strictly on the given data without inventing new operators. Variations or combinations of existing operators are acceptable.
                4. Output the response in **code format only**, avoiding any explanations, markdown, or triple backticks (` ``` `).
                5. Ensure the metaheuristic design reflects logical, diverse, and effective strategies without exceeding computational constraints.

                ### CONTEXT:
                - **Current Iteration**: {number_iteration}
                - **Best Metaheuristic So Far**: {answer}

                ### OBJECTIVE:
                Generate a code-based response that integrates the feedback and refines the metaheuristic to improve the performance.
                """
            )
        answer = output['response']

        return answer

            
