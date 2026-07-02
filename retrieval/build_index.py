import numpy as np
import faiss

print("=" * 60)
print("Loading embeddings...")
print("=" * 60)

embeddings = np.load("embeddings/shl_embeddings.npy")

print("Embedding Shape:", embeddings.shape)

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# Dimension of embedding vectors
dimension = embeddings.shape[1]

print("Vector Dimension:", dimension)

# Create FAISS Index
index = faiss.IndexFlatL2(dimension)

print("\nAdding vectors to FAISS...")

index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

# Save index
faiss.write_index(index, "retrieval/shl_index.faiss")

print("\n✅ FAISS index saved successfully!")
print("Location: retrieval/shl_index.faiss")