import chainlit as cl

from reactome.retreival_chain import initialize_retrieval_chain


@cl.on_chat_start
async def quey_llm() -> None:
    embeddings_directory: str = "embeddings/reactome"
    llm_chain = initialize_retrieval_chain(embeddings_directory, False, False)
    cl.user_session.set("llm_chain", llm_chain)

    initial_message: str = """Welcome to React-to-me your interactive chatbot for exploring Reactome!
   Ask me about biological pathways and processes"""
    await cl.Message(content=initial_message).send()


@cl.on_message
async def query_llm(message: cl.Message) -> None:
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await llm_chain.ainvoke(message.content, callbacks=[cb])
    answer: str = res["answer"]
    if cb.has_streamed_final_answer:
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer).send()
