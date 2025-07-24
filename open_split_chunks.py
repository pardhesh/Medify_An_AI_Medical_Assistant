import pickle

with open("split_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")
