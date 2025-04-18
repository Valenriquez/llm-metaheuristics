import os 
import ollama

class CollectionManager:
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def function_create_collection(self, folder_name, collection, model_embed):
        directory = os.path.join(os.path.dirname(__file__), folder_name)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):  # Check if it's a file
                content = self.read_file(file_path)
                response = ollama.embeddings(model_embed, prompt=content)
                embedding = response.get("embedding")
                if embedding:
                    collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{"hnsw:space": "cosine"} ]
                    )
                    print(f"Added {filename} to the collection -->  {collection}")
                else:
                    print(f"Warning: Empty embedding generated for {filename} for collection -->  {collection}")
            else:  # If it's not a file, skip it
                continue
        return collection

    