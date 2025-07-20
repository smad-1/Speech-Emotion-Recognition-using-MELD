from datasets import load_from_disk

# Load dataset from saved folder
dataset = load_from_disk("meld_dataset")

# Check splits
print(dataset)

# Print first example in train split
print("\nSample:")
print(dataset["train"][0])
