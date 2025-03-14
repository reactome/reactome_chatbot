from enum import StrEnum
from typing import Callable, NamedTuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import StateGraph

from agent.profiles.react_to_me import create_reacttome_graph


class ProfileName(StrEnum):
    # These should exactly match names in .config.schema.yaml
    React_to_Me = "React-to-Me"


class Profile(NamedTuple):
    name: ProfileName
    description: str
    graph_builder: Callable[[BaseChatModel, Embeddings], StateGraph]


CHAT_PROFILES: dict[str, Profile] = {
    ProfileName.React_to_Me.lower(): Profile(
        name=ProfileName.React_to_Me,
        description="An AI assistant specialized in exploring **Reactome** biological pathways and processes.",
        graph_builder=create_reacttome_graph,
    ),
}


def create_profile_graphs(
    profiles: list[ProfileName],
    llm: BaseChatModel,
    embedding: Embeddings,
) -> dict[str, StateGraph]:
    return {
        profile: CHAT_PROFILES[profile].graph_builder(llm, embedding)
        for profile in map(str.lower, profiles)
    }


def get_chat_profiles(profiles: list[ProfileName]) -> list[Profile]:
    return [CHAT_PROFILES[profile.lower()] for profile in profiles]
