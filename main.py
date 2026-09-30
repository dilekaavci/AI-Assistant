from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("phi-3.5-mini")

    print("Preparing model...")
    model.download(
        lambda progress: print(
            f"Downloading: %{progress:.1f}",
            end="",
            flush=True
        )
    )

    print("Dowloading..")
    model.load()
    print("Model is ready!")


    client = model.get_chat_client()
    messages = [
        {
            "role": "user",
            "content": "What is Retrieval-Augmented Generation?"
        }
    ]


    response = client.complete_chat(messages)

    print("AI answer:")
    print(response.choices[0].message.content)
    model.unload()


if __name__ == "__main__":
    main()