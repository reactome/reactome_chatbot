import argparse
import random
import os
import pprint as pp

from dotenv import load_dotenv

from src.retreival_chain import initialize_retrieval_chain


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Reactome ChatBot")
    parser.add_argument("--openai-key", help="API key for OpenAI")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--no-delay", action="store_true", help="No delay with outputing response"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY") or args.openai_key
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Please provide the API key using --openai-key or in a .env file."
        )

    # Set the OPENAI_API_KEY environment variable
    os.environ["OPENAI_API_KEY"] = openai_key

    embeddings_directory = "embeddings"
    qa = initialize_retrieval_chain(embeddings_directory, args.verbose)
    if args.interactive:
        await interactive_mode(qa, args.verbose, args.no_delay)
    else:
        query = "Provide a comprehensive list of all entities (including their names and IDs) where GTP is a component."
        await print_results(qa, query, args.verbose, args.no_delay)


async def interactive_mode(qa, verbose, no_delay):
    while True:
        query = input("Enter your query (or press Enter to exit): ")
        if not query:
            break
        await print_results(qa, query, verbose, no_delay)


async def print_results(qa, query, verbose, no_delay):
    async for qa_results in qa.astream(query):
        print("\nResponse:")
        answer = qa_results["answer"]
        answer = answer.strip("('").rstrip("')")
        answer = answer.replace("\\n", "\n")

        words = answer.split()
        for word in words:
            print(word, end=' ', flush=True)
            if not no_delay:
                delay_time = random.uniform(0, 0.1)
                await asyncio.sleep(delay_time)

        print("\n")
        if verbose:
            print("Chat History")
            pp.pprint(qa_results["chat_history"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
