
import os
import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="ai_documents"
)

# Document folder
document_folder = "./dataset/documents"

documents = []
ids = []
metadatas = []

# Load documents
for filename in os.listdir(document_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(document_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)
        ids.append(filename)

        metadatas.append({
            "title": filename,
            "category": "Artificial Intelligence",
            "source": file_path
        })

# Generate embeddings
embeddings = model.encode(documents)

# Store embeddings in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

# Save embeddings as NumPy file
np.save("./dataset/embeddings.npy", embeddings)

print("Documents successfully stored in ChromaDB!")
print("Total documents:", collection.count())

# Load user queries
queries_df = pd.read_csv("./dataset/queries.csv")

retrieved_results = []

# Perform similarity search
for _, row in queries_df.iterrows():

    query = row["query"]

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Retrieve Top-5 relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("\nQuery:", query)
    print("-" * 50)

    for i in range(5):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print("Rank:", i + 1)
        print("Document:", metadata["title"])
        print("Content:", document)
        print("Distance:", round(distance, 4))
        print()

        retrieved_results.append({
            "query": query,
            "rank": i + 1,
            "document": metadata["title"],
            "category": metadata["category"],
            "source": metadata["source"],
            "retrieved_text": document,
            "distance": distance
        })

# Save retrieval results
results_df = pd.DataFrame(retrieved_results)

results_df.to_csv(
    "./dataset/retrieved_results.csv",
    index=False
)

print("Retrieval results saved successfully!")

# Keyword-based search comparison

print("\nKEYWORD SEARCH COMPARISON")
print("=" * 50)

keyword_results = []

for _, row in queries_df.iterrows():

    query = row["query"].lower()
    query_words = set(query.split())

    scores = []

    for filename, document in zip(ids, documents):

        document_words = set(document.lower().split())

        # Count matching words
        score = len(query_words.intersection(document_words))

        scores.append({
            "query": query,
            "document": filename,
            "keyword_score": score
        })

    # Sort documents by keyword score
    scores = sorted(
        scores,
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    print("\nQuery:", query)

    for result in scores[:5]:

        print(
            result["document"],
            "Score:",
            result["keyword_score"]
        )

        keyword_results.append(result)

# Save keyword results
keyword_df = pd.DataFrame(keyword_results)

keyword_df.to_csv(
    "./dataset/keyword_results.csv",
    index=False
)

print("\nKeyword comparison saved successfully!")
