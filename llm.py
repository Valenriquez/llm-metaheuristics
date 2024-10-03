"""LLM manager to connect to different types of models.
"""
import ollama
 


class LLMmanager:
    """LLM manager, currently only supports ChatGPT models."""

    def __init__(self, api_key, model="deepseek-coder-v2"):
        """Initialize the LLM manager with an api key and model name.

        Args:0.8
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, llama3, codellama, deepseek-coder-v2, gemma2, codegemma,
        """
        self.model = model
         

    def chat(self, session_messages):
    # first concatenate the session messages
        big_message = ""
        for msg in session_messages:
            big_message += msg["content"] + "\n"
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": big_message,
                }
            ],
        )
        return response["message"]["content"]
