# Architecture

How RAGBase works, drawn from the code. Module names are given so each
diagram can be checked against the source.

- [Local application and public showcase](#local-application-and-public-showcase)
- [Ingestion and indexing](#ingestion-and-indexing)
- [Hybrid retrieval and evidence citations (Standard RAG)](#hybrid-retrieval-and-evidence-citations-standard-rag)
- [Agentic RAG: the LangGraph workflow](#agentic-rag-the-langgraph-workflow)
- [Claim verification and bounded refinement](#claim-verification-and-bounded-refinement)
- [Inference backends: Ollama and the optional Prism gateway](#inference-backends-ollama-and-the-optional-prism-gateway)
- [Tabular questions](#tabular-questions)

## Local application and public showcase

Two Streamlit experiences share the same rendering code and stylesheet but
not the same capabilities.

```mermaid
flowchart TB
    subgraph local["Full local app — streamlit run app.py"]
        LA["ui/local_app.py"] --> ING["ingest.py<br/>per-session workspace"]
        LA --> RET["retriever.py<br/>Standard RAG"]
        LA --> AG["agent/<br/>Agentic RAG"]
        LA --> TAB["tabular.py"]
        RET --> LLM["llm/ — Ollama or Prism"]
        AG --> LLM
        TAB --> LLM
    end
    subgraph show["Public showcase — showcase/streamlit_app.py"]
        SC["ui/showcase.py"] --> DATA[("showcase/data/*.json<br/>recorded runs, corpus text")]
    end
    LA --> R["ui/render.py — answers, citations, traces"]
    SC --> R
    REC["scripts/record_showcase.py<br/>runs the real pipeline locally"] -. writes .-> DATA
```

- The showcase imports no ingestion, retrieval-model, agent, LLM or tabular
  code and offers no text input or upload widget; it replays answers that
  `scripts/record_showcase.py` recorded from the real pipeline, with
  provenance. It is deployed from the `showcase-deploy` branch
  ([deployment details](DEPLOYMENT.md)).
- `app.py` is a thin entry point that loads one experience or the other.

## Ingestion and indexing

```mermaid
flowchart LR
    F["Uploaded files<br/>PDF, DOCX, TXT, images, CSV, XLSX"] --> SAFE["Safe file names,<br/>size and count limits"]
    SAFE --> LOAD["load_doc()"]
    LOAD --> PDF["PDF: pdfplumber,<br/>column-aware page text"]
    LOAD --> OCR["Images and text-less PDF pages:<br/>Tesseract OCR (capped)"]
    LOAD --> DOCX["DOCX / TXT"]
    LOAD --> ROWS["CSV / Excel:<br/>row blocks"]
    PDF --> SPLIT["RecursiveCharacterTextSplitter<br/>(chunk size, overlap)"]
    OCR --> SPLIT
    DOCX --> SPLIT
    ROWS --> META
    SPLIT --> META["Chunks with source, location<br/>and deterministic chunk IDs (sha256)"]
    META --> EMB["all-MiniLM-L6-v2 embeddings"] --> FAISS[("FAISS index")]
    META --> TFIDF[("TF-IDF vectorizer + matrix")]
    META --> MAN[("manifest.json<br/>embedding model, chunk count")]
```

- Two-column PDF pages are detected conservatively and read column by
  column, so side-by-side rules (for example, over-21 and under-21 limits)
  are not interleaved.
- In the local app the index is built in a temporary workspace for the
  browser session (`ui/uploads.py`).

## Hybrid retrieval and evidence citations (Standard RAG)

```mermaid
flowchart LR
    Q["Question"] --> SEM["FAISS semantic search<br/>top 20"]
    Q --> KW["TF-IDF keyword search<br/>top 20"]
    SEM --> RRF["Reciprocal-rank fusion<br/>(k = 60)"]
    KW --> RRF
    RRF --> CE["Cross-encoder reranking<br/>ms-marco-MiniLM-L-6-v2"]
    CE --> TOP["Top 5 passages"]
    TOP --> EV["Numbered evidence<br/>[1] … [5]"]
    EV --> GEN["One model call:<br/>answer with [n] citations"]
    GEN --> CHK["Deterministic citation check<br/>IDs exist · no invented references"]
    CHK --> OUT["Answer + cited passages<br/>+ retrieved-but-uncited passages"]
```

- Citation **validity** (the IDs exist) is checked for every answer. It does
  not show that the cited passage supports the claim; Standard RAG performs
  no claim verification and the interface says so.
- Evidence text is passed to the model as untrusted data.

## Agentic RAG: the LangGraph workflow

```mermaid
flowchart TD
    START((start)) --> PLAN["plan<br/>structured planner output"]
    PLAN -->|simple| STD["standard<br/>one search, one answer<br/>(claims not verified)"]
    PLAN -->|complex| RET["retrieve<br/>planned sub-queries"]
    PLAN -->|stopped| FIN
    STD --> FIN["finalize<br/>status + notes"]
    RET --> SYN["synthesize<br/>answer citing evidence IDs"]
    SYN --> VER["verify<br/>deterministic checks + claim verdicts"]
    VER -->|"revise, revisions left, time left"| REF["refine<br/>follow-up query or feedback"]
    VER -->|otherwise| FIN
    REF -->|follow-up query| RET
    REF -->|feedback only| SYN
    RET -.->|stopped| FIN
    SYN -.->|stopped| FIN
    FIN --> END((end))
```

- **Bounds:** at most one refinement round (default), a time budget, at most
  three sub-queries, and at most five documents per comparison. A failed or
  timed-out step goes straight to `finalize`.
- **Comparison questions:** `agent/sources.py` searches the documents named
  in the question (or by the planner); otherwise one relevance search picks
  the documents. Passages are interleaved by source, and any limitation
  (fewer than two documents found, or more than the limit) is recorded.
- **Tools:** read-only `hybrid_search` and `list_documents`. Unknown source
  names are dropped, so model output cannot widen the search.
- Every step emits a structured trace event (route, sub-queries, searches,
  documents chosen, verification counts), which the interface shows. The
  planner's free-text rationale is not displayed.

## Claim verification and bounded refinement

```mermaid
flowchart TD
    A["Synthesized answer"] --> ABS{"Full abstention?<br/>(no substantive clause)"}
    ABS -->|yes| AC["Abstention check:<br/>does the evidence answer it?"]
    AC --> S1["completed if confirmed · qualified if the evidence<br/>may contain an answer · not fully checked if the check failed"]
    ABS -->|no| SPLIT["Split into claims;<br/>absence statements set aside"]
    SPLIT --> DET["Deterministic checks<br/>citation IDs · numbers present in cited text"]
    SPLIT --> LLMV["Model verdict per claim<br/>supported · partially · unsupported"]
    DET --> G["Grounding:<br/>verified · partially verified · incomplete ·<br/>unsupported · not verified"]
    LLMV --> G
    G -->|"verified"| OK["completed"]
    G -->|"problems, revision allowed"| R["refine (once)"]
    G -->|"unconfirmed but no contradiction"| UV["completed — not fully checked"]
    G -->|"unsupported or number mismatch"| Q["qualified (warning shown)"]
```

- A failed verifier call is never treated as verification; the answer is
  reported as not fully checked.
- Statements that information is missing are not verified as facts; the
  answer carries a note that they are based on the passages retrieved.
- The interface describes a "verified" result as an automated check by a
  language model, not a guarantee of correctness.

## Inference backends: Ollama and the optional Prism gateway

```mermaid
flowchart LR
    APP["RAGBase<br/>llm/factory.py"] -->|default| OL["OllamaChatModel"] --> OS[("Ollama<br/>on this machine")]
    APP -->|"RAGBASE_LLM_BACKEND=prism<br/>+ PRISM_API_KEY"| PR["PrismChatModel"] --> GW["Prism gateway<br/>guardrails · 1-hour response cache"]
    GW --> PROV["Provider"]
    PROV --> CHECKP{"provider_used in<br/>PRISM_ALLOWED_PROVIDERS?"}
    CHECKP -->|yes| OKP["Response accepted"]
    CHECKP -->|"no, or cache hit<br/>without provider"| REJ["Rejected (fail closed)"]
```

- Direct Ollama is the default and keeps all text on the machine.
- Prism is offered only when configured. Its guardrails can block prompts,
  and it caches responses for an hour (so document-derived text is held in
  its cache); repeated identical requests can be rejected as cache hits. See
  [PRISM.md](PRISM.md).

## Tabular questions

```mermaid
flowchart LR
    TQ["Question about a CSV / Excel file"] --> PLANQ["Model proposes a query<br/>(structured output)"]
    PLANQ --> VAL["Validated against an allowlist:<br/>operations, columns, filters"]
    VAL -->|valid| EXEC["Executed with pandas<br/>by RAGBase code"]
    VAL -->|invalid| REFUSE["Refused"]
    EXEC --> ANS["Answer with the operation shown"]
```

Model output is never executed as code. Not available in the public showcase.
