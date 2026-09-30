import sqlite3


DB_NAME = "rag_database.db"


def create_database():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()