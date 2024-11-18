import ollama


data = r"/Users/valeriaenriquezlimon/Documents/new-td/llm-metaheuristics/parameters_to_take.txt"

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Strip whitespace and exclude empty lines
    content = [line.strip() for line in lines if line.strip()]
    return content

# Load the file content into a variable
file_content = load_txt(data)

prompt = f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
        you should only use the information provided in the collection. Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' 
        instead of typing a space. Remember that, if the dimension is 3 or bigger, you should add more than 2 metaheuristic operators with it's parameters and selector, you could add more than two metaheuristics too. 
        In the 'fun' variable you must change it too: 'fun = bf.Ackley(1)"""
# generate a response combining the prompt and data we retrieved in step 2
print("data--", file_content)
output = ollama.generate(
  model="llama3.2:latest",
  prompt=f"Using this data: {file_content}. Respond to this prompt: {prompt}"
)

print(output['response'])
