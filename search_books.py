import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load saved files
df = pd.read_csv("books_metadata.csv")
index = faiss.read_index("books.index")
model = SentenceTransformer('all-MiniLM-L6-v2')

# User input (description or query)
query = input("Enter a short description or a book title to find similar books: ")
query_embedding = model.encode([query])

# Search similar
k = 5
D, I = index.search(np.array(query_embedding), k)

print("\n📚 Top Similar Books:")
for idx in I[0]:
    print(f"- {df.iloc[idx]['title']} by {df.iloc[idx]['authors']}")
