"""The instruction that makes an answer come back in the user's language.

Shared by every answer prompt so the two deployments cannot drift apart.

The wording is @bleedblack1's from #140, which got the hard part right: the
retrieved context is English and must stay English, but scientific nomenclature
inside the answer must not be translated either. `SET`, `MAX` and `CAT` are gene
symbols and also ordinary English words, and `R-HSA-9612973` means nothing
translated.

What is not taken from #140 is where it put this. That PR appends it to `input`,
which `create_retrieval_chain` hands straight to the retriever -- verified in
`langchain_classic`: `retrieval_docs = (lambda x: x["input"]) | retriever`. Measured
through the whole retriever, that changes about half the fused documents. Here it is
a separate prompt variable, so `input` is untouched and retrieval, including the
query expansion in front of it, is byte-identical.
"""

LANGUAGE_INSTRUCTION = """Answer in {detected_language}.

The context you were given is in English because the Reactome knowledgebase is
English-only. Your answer must still be in {detected_language}.

Keep gene symbols, protein names, pathway names, Reactome identifiers (R-HSA-...)
and URLs exactly as they appear in the context. Do not translate them, even when a
symbol is also an ordinary word."""
