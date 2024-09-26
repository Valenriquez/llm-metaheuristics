import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime

sys.path.append('llm-metaheuristics/algorithm_creation')

# Define the function
fun = bf.Ackley1(2)

# Get the function name
fun_name = fun.__class__.__name__
print(fun_name)