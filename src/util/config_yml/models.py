"""Which model a deployment answers with.

The shape is @AaryanCode69's from #112 -- named fields rather than a
"provider/model" string to re-parse, which is also where `base_url` can live.
Plant Reactome needs that: it serves its embedding model from a self-hosted
OpenAI-compatible endpoint.

What is deliberately absent is an embedding model. It is derived from the bundle
that built the vectors (`agent.graph.resolve_embedding_model`), because a query
embedded with a different model than the stored vectors returns nonsense rather
than an error. Both #112 and #151 made it configurable; that is the one part of
them not taken. See specs/003-model-configuration/spec.md FR-004.
"""

from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    """The answering model. Every field is optional, so an `llm:` section may set
    only what it wants to change and inherit the rest."""

    # extra="forbid" so an unknown key is a validation error, which Config.from_yaml
    # treats as fatal. Pydantic's default is to ignore extras silently -- meaning a
    # config.yml saying `embedding_model: text-embedding-3-large` would be accepted,
    # discarded, and leave an operator believing they had set it. Refusing to start
    # is the only honest answer to a setting that cannot be honoured.
    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"

    # None means "not configured here", which leaves LLM_MODEL and then the
    # built-in default in charge. A default of "gpt-4o-mini" would instead make
    # every config.yml silently pin that model, which is the opposite of
    # FR-002's promise that an absent section changes nothing.
    model: str | None = None

    base_url: str | None = None

    # Almost always leave unset. The value a model requires is derived in
    # agent.graph.resolve_temperature from a measured table, because it is a
    # property of the model rather than a preference: the gpt-5.5/5.6 families,
    # o3 and o4-mini accept only 1.0, while gpt-5.1/5.2/5.4 accept 0.0. Setting
    # it here to something the model refuses now stops startup rather than
    # failing on a user's first question (FR-006).
    temperature: float | None = None
