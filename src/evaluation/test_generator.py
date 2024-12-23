import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a test set based on given documents and distributions.')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing example files')
    parser.add_argument('--model', type=str, default="gpt-4-turbo", help='Language model to use (default: gpt-4-turbo)')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature setting for the language model (default: 0.3)')
    parser.add_argument('--test_size', type=int, default=4, help='Number of tests to generate (default: 4)')
    parser.add_argument('--distributions', type=str, nargs='+', help='Distributions of test types (e.g., simple=0.25 reasoning=0.25 multi_context=0.25 conditional=0.25)')

    return parser.parse_args()

def save_testset(testset, filename):
    output_dir = 'testsets'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{filename}_testset.xlsx')

    # Convert testset to DataFrame for filtering
    df = pd.DataFrame(testset)

    # Filter out rows where "ground_truth" is "The answer to given question is not present in context"
    filtered_df = df[df['ground_truth'] != "The answer to given question is not present in context"]

    # Save the filtered DataFrame to an Excel file
    filtered_df.to_excel(output_path, index=False)

    print(f"Filtered testset saved to {output_path}")


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Parse command line arguments
        args = parse_arguments()

        # Setup document loader
        loader = DirectoryLoader(args.path)
        documents = loader.load()
        
        # Initialize Language Models and Embeddings
        llm = ChatOpenAI(model=args.model, temperature=args.temperature)
        embeddings = OpenAIEmbeddings()

        # Parse test type distributions
        distributions = {eval(key): float(value) for key, value in (dist.split('=') for dist in args.distributions)}

        # Setup and run test set generator
        generator = TestsetGenerator.from_langchain(generator_llm=llm, critic_llm=llm, embeddings=embeddings)
        testset = generator.generate_with_langchain_docs(documents, 
                                                         test_size=args.test_size, 
                                                         distributions=distributions)

        # Save the output
        base_filename = os.path.basename(args.path).split('.')[0]  # Assumes files have an extension
        output_dir = f"{base_filename}_testset"
        save_testset(testset, output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


## python your_script.py --path "/evaluation/example_files/" --model "gpt-4-turbo" --temperature 0.3 --test_size 4 --distributions simple=0.25 reasoning=0.25 multi_context=0.25 conditional=0.25
