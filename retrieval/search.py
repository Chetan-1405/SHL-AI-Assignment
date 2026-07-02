import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Loading FAISS index...")
print("=" * 60)

# Load processed catalog
df = pd.read_csv("data/shl_catalog_processed.csv")

# Load FAISS index
index = faiss.read_index("retrieval/shl_index.faiss")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Ready!")

while True:

    query = input("\nEnter your query (or type exit): ")

    if query.lower() == "exit":
        break

    # Convert query to embedding
    query_embedding = model.encode([query]).astype("float32")

    # Search top 5 results
    distances, indices = index.search(query_embedding, 5)

    print("\nTop Recommendations")
    print("=" * 60)

    for rank, idx in enumerate(indices[0], start=1):

        row = df.iloc[idx]

        print(f"\nRank {rank}")
        print("Assessment :", row["name"])
        print("Duration   :", row["duration"])
        print("Remote     :", row["remote"])
        print("Adaptive   :", row["adaptive"])
        print("URL        :", row["link"])