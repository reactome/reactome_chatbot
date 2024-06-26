import argparse
import os

import pyfiglet
from dotenv import load_dotenv
from typings import Any

from src.retreival_chain import initialize_retrieval_chain


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Reactome ChatBot")
    parser.add_argument("--openai-key", help="API key for OpenAI")
    parser.add_argument("--query", help="Query string to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY") or args.openai_key
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Please provide the API key using --openai-key or in a .env file."
        )

    os.environ["OPENAI_API_KEY"] = openai_key

    embeddings_directory = "embeddings"
    qa = initialize_retrieval_chain(embeddings_directory, True, args.verbose)
    if args.query:
        await print_results(qa, args.query, args.verbose)
    else:
        await interactive_mode(qa, args.verbose)


async def interactive_mode(qa: Any, verbose: bool) -> None:
    reactome_figlet = pyfiglet.figlet_format("React-to-me")
    print(reactome_figlet)
    print(
        "Reactome Chatbot instructions: After each response you will have an opportunity to ask another questions. If you are done type enter instead of a question to exit."
    )
    while True:
        query: str = input("\n\nUser Query:")
        if not query:
            break
        print("\nResponse:")
        await print_results(qa, query, verbose)


async def print_results(qa: Any, query: str, verbose: bool) -> None:
    async for qa_result in qa.invoke(query):
        pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
