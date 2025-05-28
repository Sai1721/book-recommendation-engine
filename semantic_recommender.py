import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the dataset
df = pd.read_csv("books_with_descriptions.csv")

# Keep only necessary columns
df = df[['title', 'authors', 'description']].dropna()

# Step 1: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)

# Step 2: Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 3: Save index and data for later use
faiss.write_index(index, "books.index")
df.to_csv("books_metadata.csv", index=False)
print("✅ Index and metadata saved.")
