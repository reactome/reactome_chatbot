from typing import Callable, NamedTuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import StateGraph

from agent.profiles.react_to_me import create_reacttome_graph


class Profile(NamedTuple):
    name: str
    description: str
    graph_builder: Callable[[BaseChatModel, Embeddings], StateGraph]


CHAT_PROFILES: dict[str, Profile] = {
    "React-to-Me": Profile(
        name="React-to-Me",
        description="An AI assistant specialized in exploring **Reactome** biological pathways and processes.",
        graph_builder=create_reacttome_graph,
    ),
}


def create_profile_graphs(
    profiles: list[str],
    llm: BaseChatModel,
    embedding: Embeddings,
) -> dict[str, StateGraph]:
    return {
        profile: CHAT_PROFILES[profile].graph_builder(llm, embedding)
        for profile in profiles
    }
