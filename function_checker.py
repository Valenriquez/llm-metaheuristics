import re
import os
# USING THE OPTUNA BEST PARAMETERS NOW:
def get_preferential_values(file_path):
    # Read the file contents
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code_file = file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

    # Regex patterns
    hyperparameters_pattern = r"Mejores hiperparámetros encontrados:\n({.*?})"
    performance_pattern = r"Mejor rendimiento encontrado:\n([\d.]+)"

    # Extract hyperparameters
    hyperparameters_match = re.search(hyperparameters_pattern, code_file, re.DOTALL)
    hyperparameters = eval(hyperparameters_match.group(1)) if hyperparameters_match else None

    # Extract performance
    performance_match = re.search(performance_pattern, code_file)
    performance = float(performance_match.group(1)) if performance_match else None

    return hyperparameters, performance

# Call the function with the correct file path
import os

base_path = "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results"
folder_name = "ollama_output_Rastrigin(5)_20241120_103923"
file_name = "execution_optuna_result_0.txt"

# Construct the file path
file_path = os.path.join(base_path, folder_name, file_name)
print("Constructed file path:", file_path)

# Check if the file exists
if not os.path.exists(file_path):
    print("Error: File does not exist:", file_path)
else:
    # Ensure the file is readable
    if not os.access(file_path, os.R_OK):
        print("Error: File is not readable:", file_path)
    else:
        print("File exists and is readable. Proceeding...")
        # Call the function
        hyperparameters, performance = get_preferential_values(file_path)
        print("Extracted Hyperparameters:", hyperparameters)
        print("Extracted Performance:", performance)
