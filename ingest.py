import os
import json
import sqlite3
from sentence_transformers import SentenceTransformer


DB_NAME = "rag_database.db"
DATA_FOLDER = "data"


def chunk_text(text):
    paragraphs = text.split("\n")
    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks
def save_chunk(source, chunk, embedding):
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO documents (source, chunk_text, embedding)
        VALUES (?, ?, ?)
        """,
        (
            source,
            chunk,
            json.dumps(embedding.tolist())
        )
    )

    connection.commit()
    connection.close()


def clear_database():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM documents")

    connection.commit()
    connection.close()

    print("Old records cleared .")


def main():
    print("Embedding model is loading..")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    #clear database
    clear_database()

    #find txt files
    for filename in os.listdir(DATA_FOLDER):

        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            chunks = chunk_text(text)

            print(f"\n{filename} is found.")
            print(f"{len(chunks)} chunk created.")

            # create embeddings and save to database
            for chunk in chunks:
                embedding = model.encode(chunk)

                save_chunk(
                    filename,
                    chunk,
                    embedding
                )

                print("Saved:", chunk)

    print("All documents saved to the database")


if __name__ == "__main__":
    main()