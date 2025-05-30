import csv
import json
import tensorflow_hub as hub
import tensorflow as tf

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

movies = []

with open('movies.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  

    for row in reader:
        if len(row) >= 2:
            title = row[0].strip()
            tags = ', '.join(tag.strip() for tag in row[1:])
            text = title + " " + tags  
            movies.append({
                "title": title,
                "tags": tags,
                "text": text
            })

texts = [movie["text"] for movie in movies]
embeddings = embed(texts).numpy()

for i, movie in enumerate(movies):
    movie["embedding"] = embeddings[i].tolist()
    del movie["text"]  

with open("movie_embeddings.json", "w", encoding='utf-8') as f:
    json.dump(movies, f, indent=2)

print(f"Processed {len(movies)} movies and saved embeddings to movie_embeddings.json")
