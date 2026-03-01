from typing import Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: Literal["openai", "ollama"] | str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "huggingfacehub", "huggingfacelocal"] | str = "openai"
    model: str = "text-embedding-3-large"
    device: str | None = "cpu"


class ModelsConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
