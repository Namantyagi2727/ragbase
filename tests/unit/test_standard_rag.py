"""Standard RAG: evidence IDs in the prompt, citations validated on the way out."""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import retriever as r
from citations import ABSTAIN_TEXT
from llm import LLMUnavailableError
from llm.fake import FakeChatModel


class FixedRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.docs


DOCS = [
    Document(page_content="Meal allowance is 45 dollars.",
             metadata={"chunk_id": "c1", "source": "travel.txt", "loc": "txt"}),
    Document(page_content="Hotels above 200 dollars need approval.",
             metadata={"chunk_id": "c2", "source": "travel.txt", "loc": "txt"}),
]


def _run(reply, docs=DOCS):
    model = FakeChatModel([reply])
    result = r.build_rag_chain(FixedRetriever(docs=docs), chat_model=model).invoke("What is the meal allowance?")
    return result, model


def test_prompt_includes_evidence_ids_and_source_identifiers():
    _, model = _run("The allowance is 45 dollars per day [E1].")
    prompt = "\n".join(m["content"] for m in model.calls[0]["messages"])
    assert '<doc id="E1" source="travel.txt"' in prompt
    assert '<doc id="E2" source="travel.txt"' in prompt
    assert "What is the meal allowance?" in prompt


def test_result_keeps_original_keys_and_adds_citations():
    result, _ = _run("The allowance is 45 dollars per day [E1].")
    assert result["answer"] == "The allowance is 45 dollars per day [E1]."
    assert result["context"] == DOCS
    assert [e.eid for e in result["evidence"]] == ["E1", "E2"]
    assert result["citations"].valid_ids == ["E1"]
    assert result["citations"].ok
    assert result["llm"].is_mock


def test_hallucinated_evidence_id_is_flagged():
    result, _ = _run("The allowance is 45 dollars per day [E7].")
    assert result["citations"].invalid_ids == ["E7"]
    assert not result["citations"].ok


def test_free_form_file_citation_is_flagged():
    result, _ = _run("The allowance is 45 dollars per day [Driver's Manual @ Table of Contents].")
    assert result["citations"].non_evidence_citations
    assert not result["citations"].ok


def test_abstention_is_valid_without_citations():
    result, _ = _run(ABSTAIN_TEXT, docs=[])
    assert result["citations"].abstained
    assert result["citations"].ok


def test_backend_failure_propagates_as_llm_error():
    model = FakeChatModel([LLMUnavailableError("down")])
    chain = r.build_rag_chain(FixedRetriever(docs=DOCS), chat_model=model)
    with pytest.raises(LLMUnavailableError):
        chain.invoke("q")
