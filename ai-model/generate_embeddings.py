import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

# Load the CSV
df = pd.read_csv("movies.csv")

# Prepare text
df["tag_text"] = df["tags"].astype(str).str.lower()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed tags
df["embedding"] = df["tag_text"].apply(lambda x: model.encode(x))

# Save to JSON
movie_data = []
for _, row in df.iterrows():
    movie_data.append({
        "title": row["title"],
        "tags": row["tag_text"],
        "embedding": row["embedding"].tolist()
    })

with open("movie_embeddings.json", "w") as f:
    json.dump(movie_data, f, indent=2)

print("✅ movie_embeddings.json created at:", os.getcwd())

