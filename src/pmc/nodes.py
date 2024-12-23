import os

from dotenv import load_dotenv

from conversational_chain.graph import RAGGraphWithMemory
from completeness_grader import completeness_grader
from pmc_wrapper import PMCBestMatchAPIWrapper
from question_rewriter import question_rewriter
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


def generate(state):
    """
    Generate answer

    Args:
        state(dict): The current graph state.

    Returns:
        state (dict): New key added to state, generation that contains LLM generation.

    """
    print(" ___GENERATE___")
    question = state["question"]
    generation = llm_graph.invoke(question)
    state["generation"] = generation
    return state


def grade_completeness(state):
    """
    Determine whether the LLM-generated response based on Reactome data is complete.

    Args:
        state (dict): The current state graph.

    Returns:
        state (dict): Updated state with pmc_search key indicating if PMC search is needed.

    """
    question = state["question"]
    generation = state["generation"]

    completeness = completeness_grader.invoke(
        {"question": question, "generation": generation}
    )
    state["pmc_search"] = completeness.binary_score

    return state


def transform_query(state):
    """
    Transform the query to produce an appropriate query for searching PMC.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Updates question key with a re-phrased question.
    """
    print("___TRANSFORM QUERY___")
    question = state["question"]
    pmc_question = question_rewriter.invoke({"question": question})
    state["pmc_question"] = pmc_question
    return state


def perform_pmc_search(state):
    """
    Searches the PMC using the LLM-created query.

    Args:
        state (dict): The current state graph.

    Returns:
        state (dict): Updates the pmc_search_results key with appended PMC results.

    """
    pmc_question = state["pmc_question"]
    generation = state["generation"]

    print("___PMC SEARCH___")
    email = "helia131500@gmail.com"
    wrapper = PMCBestMatchAPIWrapper(email=email)
    docs = wrapper.invoke(pmc_question, max_results=3)

    links = [f'<a href="{item["links"]}">{item["title"]}</a>' for item in docs]
    search_results_str = "\n - ".join(links)
    final = f"{generation} \n\nHere are some papers from PMC that you may find helpful:\n\n  - {search_results_str}"

    state["generation"] = final
    state["pmc_search_results"] = links
    return state


def decide_to_search_pmc(state):
    """
    Determines whether the LLM-generated response completely answers user's question.

    Args:
        state (dict): The current state graph.

    Returns:
        str: The next step in the workflow.
    """
    pmc_search = state["pmc_search"]
    print("___ASSESS ANSWER COMPLETENESS___")
    if pmc_search == "No":
        print("___RESPONSE NOT COMPLETE: SEARCH PMC___")
        return "transform_query"
    else:
        print("___RESPONSE COMPLETE___")
        return "finish"
