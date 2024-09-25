from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from system_prompt.reactome_prompt import qa_prompt, contextualize_q_prompt

class RAGChainWithMemory:
    def __init__(self, memory, retriever, llm):
        """
        Initializes the Retrieval-Augmented Generation (RAG) chain with memory.
        """
        self.memory = memory
        self.llm = llm

        # Create the history-aware retriever
        self.history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=contextualize_q_prompt,
        )

        # Create the documents chain
        self.question_answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )

        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(
            retriever=self.history_aware_retriever,
            combine_docs_chain=self.question_answer_chain,
        )

    def invoke(self, user_input):
        """
        Runs the chain synchronously.
        """
        # Invoke the chain and get the parsed output
        response = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": self.memory.get_history(),
        })

        answer = response['answer']

        # Update memory with user input and LLM response
        self.memory.add_human_message(user_input)
        self.memory.add_ai_message(answer)

        return answer

    async def ainvoke(self, user_input, callbacks=None):
        """
        Runs the chain asynchronously.
        """
        # Invoke the chain asynchronously
        response = await self.rag_chain.ainvoke({
            "input": user_input,
            "chat_history": self.memory.get_history(),
        }, callbacks=callbacks)

        answer = response['answer']

        # Update memory with user input and LLM response
        self.memory.add_human_message(user_input)
        self.memory.add_ai_message(answer)

        return answer
