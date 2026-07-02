import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Loading processed catalog...")
print("=" * 60)

df = pd.read_csv("data/shl_catalog_processed.csv")

print(f"Total Assessments : {len(df)}")

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

print("\nGenerating embeddings...")

embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding Shape:", embeddings.shape)

np.save("embeddings/shl_embeddings.npy", embeddings)

print("\nEmbeddings saved successfully!")
print("Location: embeddings/shl_embeddings.npy")