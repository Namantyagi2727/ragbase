"""
retriever.py — LangChain-based retrieval pipeline for RAGBase.

Components:
  HybridRetriever      — FAISS semantic + TF-IDF keyword search combined
  CrossEncoderReranker — optional re-ranking wrapper
  build_rag_chain()    — assembles the RetrievalQA chain
  answer_tabular()     — pandas code-gen path for numeric/tabular queries
  get_retriever()      — factory used by app.py (@st.cache_resource)
"""

import os
import re
import pickle
import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(model: str = config.OLLAMA_MODEL) -> OllamaLLM:
    return OllamaLLM(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=4096,
    )


# ---------------------------------------------------------------------------
# Index loading helpers
# ---------------------------------------------------------------------------

def load_vectorstore(
    index_dir: str = config.INDEX_DIR,
    embed_model: str = config.EMBED_MODEL,
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_tfidf(index_dir: str = config.INDEX_DIR) -> tuple:
    """Returns (vectorizer, tfidf_matrix, chunks_meta)."""
    with open(os.path.join(index_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(index_dir, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(index_dir, "chunks_meta.pkl"), "rb") as f:
        chunks_meta = pickle.load(f)
    return vectorizer, tfidf_matrix, chunks_meta


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """Combines FAISS semantic search and TF-IDF keyword search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    vectorizer: Any
    tfidf_matrix: Any
    chunks_meta: List[dict]
    top_k: int = config.TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # Semantic search via FAISS
        faiss_docs = self.vectorstore.similarity_search(query, k=self.top_k)

        # Keyword search via TF-IDF
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(self.tfidf_matrix, qv).flatten()
        top_indices = np.argsort(sims)[::-1][: self.top_k]
        tfidf_docs = [
            Document(
                page_content=self.chunks_meta[i]["text"],
                metadata={
                    "source": self.chunks_meta[i]["source"],
                    "loc": self.chunks_meta[i]["loc"],
                },
            )
            for i in top_indices
            if sims[i] > 0
        ]

        # Merge and deduplicate
        seen: set = set()
        merged: List[Document] = []
        for doc in faiss_docs + tfidf_docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("loc", ""),
                doc.page_content[:80],
            )
            if key not in seen:
                seen.add(key)
                merged.append(doc)
            if len(merged) >= self.top_k:
                break

        return merged


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseRetriever):
    """Re-ranks HybridRetriever results using a cross-encoder model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: HybridRetriever
    top_k: int = config.TOP_K
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        candidates = self.base_retriever._get_relevant_documents(query)
        if len(candidates) <= 1:
            return candidates

        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(self.model_name)
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = ce.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[: self.top_k]]
        except Exception as e:
            log.warning("Reranking failed (%s), using unranked results.", e)
            return candidates[: self.top_k]


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful document Q&A assistant.
Answer the question using ONLY the information in the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Provide a clear, concise answer. After your answer, list the sources you used under a "Sources:" heading, \
citing each as [filename @ location].
""",
)


def build_rag_chain(retriever: BaseRetriever, llm_model: str = config.OLLAMA_MODEL):
    llm = get_llm(llm_model)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    }).assign(answer=chain_from_docs)


# ---------------------------------------------------------------------------
# Tabular QA
# ---------------------------------------------------------------------------

def _load_dataframes(docs_dir: str) -> dict:
    dfs: dict = {}
    if not os.path.isdir(docs_dir):
        return dfs
    for fn in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fn)
        ext = fn.lower().rsplit(".", 1)[-1]
        if ext == "csv":
            df = pd.read_csv(path, low_memory=False).infer_objects()
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            dfs[fn] = df
        elif ext in ("xlsx", "xls"):
            xls = pd.ExcelFile(path)
            sheets: dict = {}
            for sh in xls.sheet_names:
                df = xls.parse(sh).infer_objects()
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                sheets[sh] = df
            dfs[fn] = sheets
    return dfs


def is_tabular_question(question: str, docs_dir: str) -> tuple[bool, dict]:
    """Returns (is_tabular, dataframes_dict)."""
    dfs = _load_dataframes(docs_dir)
    q_lower = question.lower()
    filename_hit = any(fn.lower() in q_lower for fn in dfs)
    numeric_hit = any(op in q_lower for op in config.NUMERIC_OPS)
    return (filename_hit or numeric_hit) and bool(dfs), dfs


def answer_tabular(question: str, dfs: dict, llm_model: str = config.OLLAMA_MODEL) -> str:
    """Generate and execute pandas code to answer a numeric/tabular question."""
    # Pick the most relevant dataframe
    q_lower = question.lower()
    selected_df = None
    for fn, obj in dfs.items():
        if fn.lower() in q_lower:
            selected_df = obj
            break
    if selected_df is None:
        selected_df = next(iter(dfs.values()))

    if isinstance(selected_df, dict):
        _, df = next(iter(selected_df.items()))
    else:
        df = selected_df

    cols = df.columns.tolist()
    llm = get_llm(llm_model)
    code_prompt = (
        f"You are a pandas expert. A DataFrame `df` has columns: {cols}.\n"
        f"Write Python code (using `df`) to answer: {question}\n"
        f"Assign the final answer to a variable named `ans` and print(ans).\n"
        f"Return ONLY the code, no explanations."
    )
    raw_code = llm.invoke(code_prompt)
    # Strip markdown fences
    raw_code = re.sub(r"^```(?:python)?", "", raw_code.strip(), flags=re.MULTILINE)
    raw_code = re.sub(r"```$", "", raw_code.strip(), flags=re.MULTILINE)
    # Remove any file-read lines for safety
    safe_lines = [ln for ln in raw_code.splitlines() if "read_" not in ln]
    code = "\n".join(safe_lines)

    local_ns: dict = {"df": df, "pd": pd, "ans": None}
    try:
        exec(code, {}, local_ns)  # nosec - controlled internal use only
        return str(local_ns.get("ans", "No result produced."))
    except Exception as e:
        return f"Error executing generated code: {e}\n\nGenerated code:\n{code}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever(
    index_dir: str = config.INDEX_DIR,
    embed_model: str = config.EMBED_MODEL,
    top_k: int = config.TOP_K,
    reranking: bool = config.RERANKING_ENABLED,
) -> BaseRetriever:
    """Load indexes and assemble the full retriever stack."""
    vectorstore = load_vectorstore(index_dir, embed_model)
    vectorizer, tfidf_matrix, chunks_meta = load_tfidf(index_dir)

    hybrid = HybridRetriever(
        vectorstore=vectorstore,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        chunks_meta=chunks_meta,
        top_k=top_k,
    )

    if reranking:
        return CrossEncoderReranker(base_retriever=hybrid, top_k=top_k)
    return hybrid
