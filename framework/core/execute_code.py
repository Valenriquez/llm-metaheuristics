from dataclasses import dataclass
import os
import subprocess

@dataclass(frozen=True) # prevents mutation
class ExecutionConfig:
    output_dir: str
    
class CodeExecutor:
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def execute_generated_code(self, code: str, number_iteration: int, is_optuna: bool = False) -> str:
        prefix = "execution_optuna_" if is_optuna else "execution_"
        file_name = os.path.join(self.config.output_dir, f'{prefix}iteration_{number_iteration}.py')

        # Write the code to a file
        with open(file_name, 'w') as f:
            f.write(code)

        try:
            # Execute the code
            result = subprocess.run(
                ['python', file_name],
                capture_output=True,
                text=True,
                timeout=140
            )

            # Collect result
            execution_result = (
                f"Exit code: {result.returncode}\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )

            # Write results to a result file
            result_file_name = f'{prefix}result_{number_iteration}.txt'
            result_file_path = os.path.join(self.config.output_dir, result_file_name)
            file_result = result.returncode # Either 1 or 0

            with open(result_file_path, 'w') as f:
                f.write(execution_result)

            return file_result # Either 1 or 0

        except subprocess.TimeoutExpired:
            return "Execution timed out after 140 seconds"
        except Exception as e:
            return f"An error occurred during execution: {str(e)}"
