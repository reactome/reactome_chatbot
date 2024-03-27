import argparse
import os
import pprint as pp

from dotenv import load_dotenv

from src.retreival_chain import initialize_retrieval_chain


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Reactome ChatBot")
    parser.add_argument("--openai-key", help="API key for OpenAI")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
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
        interactive_mode(qa, args.verbose)
    else:
        query = "Provide a comprehensive list of all entities (including their names and IDs) where GTP is a component."
        print_results(qa, query, args.verbose)


def interactive_mode(qa, verbose):
    while True:
        query = input("Enter your query (or press Enter to exit): ")
        if not query:
            break
        print_results(qa, query, verbose)


def print_results(qa, query, verbose):
    qa_results = qa.invoke(query)
    pretty_print_results(qa_results, verbose)


def pretty_print_results(qa_results, verbose):
    print("Response")
    answer = qa_results["answer"]
    # Remove the outer parentheses
    answer = answer.strip("('").rstrip("')")
    # Replace '\n' with actual new lines
    answer = answer.replace("\\n", "\n")
    print(answer)
    if verbose:
        print("Chat History")
        pp.pprint(qa_results["chat_history"])


if __name__ == "__main__":
    main()
