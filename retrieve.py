import json
import sqlite3

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DB_NAME = "rag_database.db"


def get_documents():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT source, chunk_text, embedding
        FROM documents
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


def retrieve(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(
        query,
        show_progress_bar=False
    )

    documents = get_documents()

    results = []

    for source, chunk_text, embedding_text in documents:

        chunk_embedding = json.loads(embedding_text)

        similarity = cosine_similarity(
            [query_embedding],
            [chunk_embedding]
        )[0][0]

        results.append({
            "source": source,
            "chunk": chunk_text,
            "score": similarity
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


def main():

    query = "Where is Bilkent University located?"

    results = retrieve(query)

    print("\nQuestion:")
    print(query)

    print("\nResults:")

    for result in results:

        print("\nScore:", result["score"])
        print("Source:", result["source"])
        print("Chunk:", result["chunk"])


if __name__ == "__main__":
    main()