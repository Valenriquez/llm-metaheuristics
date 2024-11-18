# rag_pipeline.py

# Import necessary libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Step 1: Load documents from a webpage
url = os.getenv('WEBPAGE_URL', "https://www.littleamerica.co.uk/blog/9-sunny-spots-in-hawaii-you-cant-miss")
loader = WebBaseLoader(url)
documents = loader.load()

# Step 2: Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(documents)

# Step 3: Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Create a vector store with the chunked documents
vectorstore = FAISS.from_documents(splitted_docs, embeddings)

# Step 5: Initialize the language model (GPT-3.5)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Step 6: Create the RetrievalQA Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Choose the appropriate chain type for your use case
    retriever=vectorstore.as_retriever()
)

# Step 7: Run the pipeline with a query
query = "places to visit in hawaii?"
result = rag_chain.invoke({"query": query})

# Step 8: Print the result
print(f"Query: {query}")
print(f"Answer: {result['result']}")