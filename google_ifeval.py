from datasets import load_dataset

# Load the official IFEval prompt dataset
dataset = load_dataset("google/IFEval")
print(dataset['train'][0])