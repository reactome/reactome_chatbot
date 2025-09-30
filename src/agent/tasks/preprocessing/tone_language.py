"""
Tone and language detection task for Graph-RAG workflow.
"""

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable


class ToneLanguageParser(BaseModel):
    language: str = Field(description="language the question is written in")
    complexity: str = Field(description="complexity of the question. could be 'simple' or 'complex'.")
    user_tone: str = Field(description="user's expertise/tone. could be 'expert' or 'non-expert'.")


def create_tone_language_detector(llm: BaseChatModel) -> Runnable:
    """
    Create a tone and language detection chain.
    
    Args:
        llm: Language model to use
        
    Returns:
        Runnable that takes user_input and returns ToneLanguageResult
    """
    system_prompt = """
You are a language and query analysis assistant. Given a user's input question, your task is to extract three distinct attributes.

1. Identify the **language** the question is written in (e.g., "English", "French", "German", "Spanish", etc.).

2. Detect the **user's tone**, based on how the question is written:
    - "expert": uses technical or field-specific terminology, assumes specialized knowledge.
    - "non-expert": uses plain or general language, suggests the user is unfamiliar with the topic.

    * Tone is based on *how* the question is phrased, not what is being asked.

3. Determine the **complexity** of the question:
    - "simple": straightforward, fact-seeking, or short; no sub-questions or dependencies.
    - "complex": includes multiple parts, nested logic, or requires synthesis/reasoning.

    * Complexity is about the *structure* and *scope* of the question, not the user's tone.

4. Output Format:
    - Do **not** answer the question.
    - Do **not** explain your reasoning.
    - Return **only** the following JSON object format:
    
        "language": "...",
        "complexity": "simple" or "complex"
        "user_tone": "expert" or "non-expert"

Here are a few examples:

Example 1:
Input: "What's the function of BRCA1 in DNA repair?"
Output:

  "language": "English",
  "user_tone": "expert",
  "complexity": "simple"


Example 2:
Input: "Can you explain how CRISPR works in gene editing?"
Output:

  "language": "English",
  "user_tone": "non-expert",
  "complexity": "simple"


Example 3:
Input: "How do environmental mutagens influence gene expression, and are those changes heritable across generations?"
Output:

  "language": "English",
  "user_tone": "non-expert",
  "complexity": "complex"
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    return prompt | llm.with_structured_output(ToneLanguageParser)
