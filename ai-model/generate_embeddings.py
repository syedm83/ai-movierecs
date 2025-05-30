import csv
import json
import tensorflow_hub as hub
import tensorflow as tf

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

movies = []

# Read your CSV file, skip the header, and prepare text for embedding
with open('movies.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # skip header

    for row in reader:
        if len(row) >= 2:
            title = row[0].strip()
            tags = ', '.join(tag.strip() for tag in row[1:])
            text = title + " " + tags  # combine title and tags for embedding
            movies.append({
                "title": title,
                "tags": tags,
                "text": text
            })

# Get embeddings for all movies
texts = [movie["text"] for movie in movies]
embeddings = embed(texts).numpy()

# Attach embeddings to the movie dictionaries and remove 'text' key
for i, movie in enumerate(movies):
    movie["embedding"] = embeddings[i].tolist()
    del movie["text"]  # remove text field as it's no longer needed

# Save as JSON
with open("movie_embeddings.json", "w", encoding='utf-8') as f:
    json.dump(movies, f, indent=2)

print(f"Processed {len(movies)} movies and saved embeddings to movie_embeddings.json")
