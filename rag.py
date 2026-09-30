import os

os.environ["HF_HUB_VERBOSITY"] = "error"

from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieve import retrieve
def answer_query(query):
    results = retrieve(query, top_k=3)

    context = "\n".join(
        result["chunk"] for result in results
    )

    print("Context:\n")
    print(context)

    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance
    model = manager.catalog.get_model("phi-3.5-mini")

    model.download()
    model.load()

    client = model.get_chat_client()

    prompt = f"""
You are a question-answering assistant for university students.

Answer the user's question using ONLY the information provided in the context below.

Rules:
1. Do not use outside knowledge.
2. If the context contains enough information to answer the question, give a clear and concise answer.
3. If the context does not contain enough information to answer the question, respond exactly with:
"I don't know based on the provided documents."
4. Do not include the "I don't know" sentence if the question can be answered from the context.
5. Do not invent missing details.

Context:
{context}

Question:
{query}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.complete_chat(messages)
    answer = response.choices[0].message.content
    model.unload()
    return answer, results


def main():
    print("- Local RAG AI Assistant -")

    query = input("Write your question: ")

    answer, sources = answer_query(query)

    print("\nAI Answer:\n")
    print(answer)

    print("\nSources:\n")

    for result in sources:
        print(
            f'{result["source"]} '
            f'(similarity: {result["score"]:.3f})'
        )


if __name__ == "__main__":
    main()
