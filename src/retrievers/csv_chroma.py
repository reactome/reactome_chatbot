from pathlib import Path
from typing import Annotated, Any, Coroutine, TypedDict

import chromadb.config
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.prompt import PromptTemplate
from nltk.tokenize import word_tokenize
from pydantic import AfterValidator, Field
from pydantic.json_schema import SkipJsonSchema

chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a biomedical question expansion engine for information retrieval over the Reactome biological pathway database.

Given a single user question, generate **exactly 4** alternate standalone questions. These should be:

- Semantically related to the original question.
- Lexically diverse to improve retrieval via vector search and RAG-fusion.
- Biologically enriched with inferred or associated details.

Your goal is to improve recall of relevant documents by expanding the original query using:
- Synonymous gene/protein names (e.g., EGFR, ErbB1, HER1)
- Pathway or process-level context (e.g., signal transduction, apoptosis)
- Known diseases, phenotypes, or biological functions
- Cellular localization (e.g., nucleus, cytoplasm, membrane)
- Upstream/downstream molecular interactions

Rules:
- Each question must be **fully standalone** (no "this"/"it").
- Do not change the core intent—preserve the user's informational goal.
- Use appropriate biological terminology and Reactome-relevant concepts.
- Vary the **phrasing**, **focus**, or **biological angle** of each question.
- If the input is ambiguous, infer a biologically meaningful interpretation.

Output:
Return only the 4 alternative questions separated by newlines.
Do not include any explanations or metadata.

Original Question: {question}""",
)


ExcludedField = SkipJsonSchema[
    Annotated[Any, Field(default=None, exclude=True), AfterValidator(lambda x: None)]
]


def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories


def create_bm25_chroma_ensemble_retriever(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    descriptions_info: dict[str, str],
    field_info: dict[str, list[AttributeInfo]],
) -> MergerRetriever:
    return HybridRetriever.from_subdirectory(
        llm,
        embedding,
        embeddings_directory,
        descriptions_info=descriptions_info,
        field_info=field_info,
        include_original=True,
    )


class RetrieverDict(TypedDict):
    bm25: BM25Retriever
    vector: SelfQueryRetriever


class HybridRetriever(MultiQueryRetriever, EnsembleRetriever):
    retriever: ExcludedField = None
    retrievers: ExcludedField = None
    _retrievers: dict[str, RetrieverDict]

    @classmethod
    def from_subdirectory(
        cls,
        llm: BaseChatModel,
        embedding: Embeddings,
        embeddings_directory: Path,
        *,
        descriptions_info: dict[str, str],
        field_info: dict[str, list[AttributeInfo]],
        include_original=False,
    ):
        _retrievers: dict[str, RetrieverDict] = {}
        for subdirectory in list_chroma_subdirectories(embeddings_directory):
            # set up BM25 retriever
            csv_file_name = subdirectory + ".csv"
            reactome_csvs_dir: Path = embeddings_directory / "csv_files"
            loader = CSVLoader(file_path=reactome_csvs_dir / csv_file_name)
            data = loader.load()
            bm25_retriever = BM25Retriever.from_documents(
                data,
                preprocess_func=lambda text: word_tokenize(
                    text.casefold(), language="english"
                ),
            )
            bm25_retriever.k = 10

            # set up vectorstore SelfQuery retriever
            vectordb = Chroma(
                persist_directory=str(embeddings_directory / subdirectory),
                embedding_function=embedding,
                client_settings=chroma_settings,
            )

            selfq_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vectordb,
                document_contents=descriptions_info[subdirectory],
                metadata_field_info=field_info[subdirectory],
                search_kwargs={"k": 10},
            )

            _retrievers[subdirectory] = {
                "bm25": bm25_retriever,
                "vector": selfq_retriever,
            }
        llm_chain = (
            super()
            .from_llm(bm25_retriever, llm, multi_query_prompt, None, include_original)
            .llm_chain
        )
        return cls(
            _retrievers=_retrievers,
            llm_chain=llm_chain,
            include_original=include_original,
            weights=[0.2] * 5,
        )

    def retrieve_documents(self, queries: list[str], run_manager) -> list[Document]:
        subdirectory_docs: list[Document] = []
        for subdirectory, retrievers in self._retrievers.items():
            bm25_retriever = retrievers["bm25"]
            vector_retriever = retrievers["vector"]
            doc_lists: list[list[Document]] = []
            for i, query in enumerate(queries):
                bm25_docs = bm25_retriever.invoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-bm25-{i}"
                        )
                    },
                )
                vector_docs = vector_retriever.invoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-vector-{i}"
                        )
                    },
                )
                doc_lists.append(bm25_docs + vector_docs)
            subdirectory_docs.extend(self.weighted_reciprocal_rank(doc_lists))
        return subdirectory_docs

    async def aretrieve_documents(
        self, queries: list[str], run_manager
    ) -> list[Document]:
        subdirectory_results: dict[
            str,
            list[
                tuple[
                    Coroutine[Any, Any, list[Document]],
                    Coroutine[Any, Any, list[Document]],
                ]
            ],
        ] = {}
        for subdirectory, retrievers in self._retrievers.items():
            bm25_retriever = retrievers["bm25"]
            vector_retriever = retrievers["vector"]
            subdirectory_results[subdirectory] = []
            for i, query in enumerate(queries):
                bm25_results = bm25_retriever.ainvoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-bm25-{i}"
                        )
                    },
                )
                vector_results = vector_retriever.ainvoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-vector-{i}"
                        )
                    },
                )
                subdirectory_results[subdirectory].append(
                    (bm25_results, vector_results)
                )
        subdirectory_docs: list[Document] = []
        for subdir_results in subdirectory_results.values():
            doc_lists: list[list[Document]] = []
            for bm25_results, vector_results in subdir_results:
                bm25_docs = await bm25_results
                vector_docs = await vector_results
                doc_lists.append(bm25_docs + vector_docs)
            subdirectory_docs.extend(self.weighted_reciprocal_rank(doc_lists))
        return subdirectory_docs
