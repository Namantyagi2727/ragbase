"""
ui/methodology.py — How RAGBase was evaluated, and the limits of that evaluation.

Deliberately contains no numerical results: their presentation is approved
separately. The qualitative findings below are the reviewed conclusions.
"""

from __future__ import annotations

TEXT = """
### What was compared

| System | What it is |
|---|---|
| **Original RAGBase** | The pre-upgrade code, run unchanged from its own commit in an isolated interpreter |
| **Standard RAG** | One hybrid search (FAISS + TF-IDF, reciprocal-rank fusion, cross-encoder reranking), one model call, numbered evidence citations checked for validity |
| **Agentic RAG** | A bounded LangGraph workflow: plan → retrieve (with comparison-aware document selection) → write → verify claims against their citations → optional refinement |

All three used the same synthetic corpus (eight policy documents and a CSV),
the same local model (`llama3.2` 3B via Ollama), the same retrieval
configuration and fixed seeds, over 42 human-reviewed questions with three
repeats each.

### What was measured

- **Correctness (keyword proxy):** required facts present and known-wrong
  statements absent, checked against human-reviewed reference facts. A proxy,
  not semantic truth.
- **Abstention:** declining unanswerable questions, and not declining answerable ones.
- **Citations:** whether cited IDs exist, and whether the cited passages
  contain the expected evidence from the expected document — measured separately.
- **Retrieval:** whether the expected passages were found, per mode.
- **Cost:** model calls, retrieval calls, tokens and latency.
- **Verifier challenge set:** fixed answer/evidence pairs with known labels,
  to measure the claim checker on its own.
- **Tabular security suite:** data questions and unsafe inputs, checked
  through sandboxed file state and blocked execution.
- **LLM judge:** a larger local model grading a subset, used as supporting
  evidence only and checked against hand-written calibration answers.

### What was found

- Under the tested configuration, **Standard RAG outperformed Agentic RAG
  overall**, at a fraction of the model calls and latency.
- The benchmark exposed two genuine agent defects — comparison questions
  searched documents in alphabetical order, and partly hedged answers were
  treated as refusals — which were fixed and re-measured in separately
  labelled runs, with the earlier results preserved.
- Scoring changes were versioned and reported separately from application
  changes.

### Limitations

- Multi-document synthesis and two-step questions remain weak in both modes.
- The claim verifier rejects some supported claims and accepts some incorrect ones.
- The tabular feature answered most, not all, functional cases; no security
  violation was observed.
- The keyword proxy and the LLM judge are both fallible.
- One small model, one machine, 42 questions: results are specific to this
  benchmark and are not a general accuracy or reliability guarantee.

Numerical results will be published once their presentation has been reviewed.
"""


def render() -> None:
    import streamlit as st

    st.markdown(TEXT)
