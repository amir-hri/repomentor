import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


chunks_path = Path("data/chunks.jsonl")
embeddings_path = Path("data/embeddings.npy")

chunks = []

with chunks_path.open("r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

texts = [chunk["content"] for chunk in chunks]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

np.save(embeddings_path, embeddings)

print(f"Chunks embedded: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Saved to: {embeddings_path}")