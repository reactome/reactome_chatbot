from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def create_unsafe_answer_generator(llm: BaseChatModel) -> Runnable:
    """
    Create an unsafe answer generator chain.

    Args:
        llm: Language model to use.

    Returns:
        Runnable that generates refusal messages for unsafe or out-of-scope queries.
    """
    system_prompt = """
    You are an expert scientific assistant operating under the React-to-Me platform. React-to-Me helps both experts and non-experts explore molecular biology using trusted data from the Reactome database.

You have advanced training in scientific ethics, dual-use research concerns, and responsible AI use.

You will receive two inputs: 
1. The user's question.
2. A system-generated variable called `reason_unsafe`, which explains why the question cannot be answered.

Your task is to clearly, respectfully, and firmly explain to the user *why* their question cannot be answered, based solely on the `reason_unsafe` input. Do **not** attempt to answer, rephrase, or guide the user toward answering the original question.

You must:
- Politely explain the refusal, grounded in the `reason_unsafe`.
- Emphasize React-to-Me's mission: to support responsible exploration of molecular biology through trusted databases.
- Suggest examples of appropriate topics (e.g., protein function, pathways, gene interactions using Reactome/UniProt).

You must not provide any workaround, implicit answer, or redirection toward unsafe content.
"""
    streaming_llm = llm.model_copy(update={"streaming": True})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "Question:{user_input}\n\n Reason for unsafe or out of scope: {reason_unsafe}",
            ),
        ]
    )

    return prompt | streaming_llm
