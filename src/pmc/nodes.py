import os

from dotenv import load_dotenv

from conversational_chain.graph import RAGGraphWithMemory
from src.pmc.query_handlers.response_evaluator import completeness_grader
from src.pmc.query_handlers.pmc_query_optimizer import pmc_question_rewriter
from src.pmc.query_handlers.resource_ranker import resource_ranker
from pmc_wrapper import PMCBestMatchAPIWrapper
from src.pmc.tavily_wrapper import TavilyWrapper
from retreival_chain import create_retrieval_chain
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging

load_dotenv()

ENV = os.getenv("CHAT_ENV", "reactome")
logging.info(f"Selected environment: {ENV}")


llm_graph: RAGGraphWithMemory = create_retrieval_chain(
    ENV,
    EmbeddingEnvironment.get_dir(ENV),
    False,
    False,
    hf_model=EmbeddingEnvironment.get_model(ENV),
)


def generate_response(state):
    """
    Generates a response using the chatbot based on the current state.

    Args:
        state (dict): The current graph state containing the user's question.

    Returns:
        dict: Updated state with the generated response added under the key 'generation'.
    """
    print("___GENERATE RESPONSE___")

    question = state["question"]
    generation = llm_graph.invoke(question)

    return {'generation':generation}


def assess_completeness(state):
    """
    Assesses whether the chatbot's response is complete and determines the need for external searches.

    Args:
        state (dict): The current graph state containing the question and generation.

    Returns:
        dict: Updated state with 'external_search' key indicating if further search is required.
    """
    question = state['question']
    generation = state['generation']
    
    completeness = completeness_grader.invoke({"question": question, "generation": generation})
    external_search = completeness.binary_score
    
    return {'external_search':external_search}


def transform_query_for_pmc(state):
    """
    Transforms the user's question into a format suitable for searching PMC.

    Args:
        state (dict): The current graph state containing the user's question.

    Returns:
        dict: Updated state with the transformed question under the key 'pmc_question'.
    """
    print("___TRANSFORM QUERY FOR PMC___")

    question = state['question']
    pmc_question = pmc_question_rewriter.invoke({"question": question})

    return {'pmc_question':pmc_question}


def perform_pmc_search(state):
    """
    Performs a search in the PMC database using the transformed query.

    Args:
        state (dict): The current graph state containing the transformed PMC query.

    Returns:
        dict: Updated state with PMC search results under the key 'pmc_search_results'.
    """
    print("___PMC SEARCH___")
        
    pmc_question = state["pmc_question"]
    
    email = 'help@reactome.org'
    wrapper = PMCBestMatchAPIWrapper(email=email)
    query = f"{pmc_question} AND review[pt]"
    pmc_search_results = wrapper.invoke(query, max_results=3)
    print('perform pmc search pmc_search_results:', pmc_search_results)
    return {'pmc_search_results':pmc_search_results}


def perform_web_search(state):
    """
    Performs a web search using the original question.

    Args:
        state (dict): The current graph state containing the user's question.

    Returns:
        dict: Updated state with web search results under the key 'web_search_results'.
    """
    print("___WEB SEARCH___")
    
    question = state['question']
    
    wrapper = TavilyWrapper(max_results=3)
    web_search_results = wrapper.invoke(question)

    return {'web_search_results':web_search_results}


def format_external_results(state):
    """
    Organizes the outputs from Web and PMC searches into a cohesive format.

    Args:
        state (dict): The current graph state containing search results and the chatbot response.

    Returns:
        dict: Updated state with the final organized output under the key 'generation'.
    """
    
    question = state['question']
    generation = state['generation']
    web_search_results = state['web_search_results']
    pmc_search_results = state['pmc_search_results']

    organized_output = resource_ranker.invoke({
        "question": question,
        "pmc_search_results": pmc_search_results,
        "tavily_search_results": web_search_results
    })
    generation = f"{generation}\n{organized_output}"

    return {'generation':generation}


### Decision Edges
def decide_next_steps(state):
    """
    Determines the next steps in the workflow based on the completeness of the chatbot's response.

    Args:
        state (dict): The current graph state.

    Returns:
        list or str: The next steps to take, either as a single action or a list of actions.
    """
    external_search = state['external_search']
    if external_search == "Yes":
        return ["perform_web_search", "transform_query_for_pmc"]
    else:
        return "finish"
