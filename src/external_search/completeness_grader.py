from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

completeness_grader_message = """
You are an expert grader with extensive knowledge in molecular biology and experience as a Reactome curator.
Your task is to evaluate whether a response generated by an LLM is complete, meaning it fully addresses the user’s query with all necessary details, background information, and context.

Additionally, assess whether the question is appropriate and directly related to biology or molecular biology.

Based on this evaluation, determine whether an external search should be conducted.

Provide a binary output as either:

Yes: The response is incomplete, missing key details, or lacking sufficient context, AND the question is appropriate and directly related to biology or molecular biology, therefore external search should be conducted.
No: Either the response is complete (fully answers the query, provides enough background, and leaves no essential details missing), OR the question is inappropriate, harmful, or not related to biology or molecular biology, therefore no external search should be conducted.
Ensure your evaluation is based solely on the information requested in the query, the adequacy of the response, and the appropriateness of the question.
"""

completeness_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", completeness_grader_message),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


class GradeCompleteness(BaseModel):
    external_search: str = Field(
        description="Answer is complete and provides all necessary background, 'Yes' or 'No'"
    )


class CompletenessGrader:
    def __init__(self, llm: BaseChatModel):
        structured_completeness_grader: Runnable = llm.with_structured_output(
            GradeCompleteness
        )
        self.runnable: Runnable = completeness_prompt | structured_completeness_grader
