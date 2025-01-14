from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


organizer_message = """
You are an expert researcher and data organizer. Your task is to process a set of links from PubMed Central and Tavily, organizing them based on their content and relevance to the user's query. 
Your goal is to make the output easy to navigate. Follow these steps:

    1. Analyze the Query: Understand the user's query and context to determine what kind of information is most relevant and prioritize relevance based on their needs.
    2. Rank links: 
        - Rank links based on their relevance to user's question.
    3. Format the Output: Use this structure:
        - Start with a friendly introduction like, "Here are some resources you may find helpful:"
        - List the links in order of descending relevance according to the following format:
            - <a href=link>title</a>
"""

organizer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", organizer_message),
        ("human",  "User question: \n\n {question} \n\n links from PubMed Central: {pmc_search_results}, links from Tavily {tavily_search_results}"),
         ]
)

resource_ranker = organizer_prompt | llm | StrOutputParser()