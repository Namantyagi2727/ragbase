"""
evals/dataset.py — Benchmark case schemas, loaders and consistency checks.

The checks here (schema, unique IDs, sources exist, evidence quotes appear in
the extracted document text, tabular expectations match the data) catch label
rot and typos. They do NOT show that a proposed answer is correct; labels
become ground truth only after human review sets review_status to "approved".
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EVALS = Path(__file__).resolve().parent
CORPUS_DIR = EVALS / "corpus"
TABULAR_DIR = EVALS / "tabular"
CASES_FILE = EVALS / "cases.jsonl"
VERIFIER_FILE = EVALS / "verifier_cases.jsonl"
TABULAR_FILE = EVALS / "tabular_cases.jsonl"

ReviewStatus = Literal["proposed", "approved", "rejected"]
Category = Literal[
    "direct_factual", "multi_document", "version_comparison", "conflicting_information",
    "unanswerable", "partially_answerable", "citation_precision", "false_premise",
    "two_step", "regression",
]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    quote: str = Field(min_length=10)


class ExpectedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    statement: str
    # Keyword proxy: every group must match at least one of its alternatives.
    keywords: list[list[str]] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class ForbiddenFact(BaseModel):
    """A known wrong answer, detected by a regex.

    Scoring rules: the pattern is matched case-insensitively against each
    sentence of the answer separately; sentences containing a negation or
    rejection cue (not, no, never, cannot, incorrect, false, rather than,
    instead of, ...) are skipped, so mentioning a wrong value in order to
    reject it is not penalised. A match is a proxy signal for review, not proof.
    """

    model_config = ConfigDict(extra="forbid")
    statement: str
    pattern: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Category
    question: str
    answerable: Literal["yes", "no", "partial"]
    expected_facts: list[ExpectedFact] = Field(default_factory=list)
    forbidden_facts: list[ForbiddenFact] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    reference_answer: str
    required: bool = False
    tags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "proposed"
    notes: str = ""


class ChallengeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source: str
    text: str


class ChallengeClaim(BaseModel):
    """One claim with two independent labels.

    label: what the verifier should conclude — is the claim supported by the
        evidence it *cites*? (citation faithfulness)
    factually_true: is the claim true according to the corpus, regardless of
        what it cites? (factual correctness)
    A true claim with the wrong citation is factually_true=True, label="unsupported".
    """

    model_config = ConfigDict(extra="forbid")
    text: str
    label: Literal["supported", "partially_supported", "unsupported"]
    factually_true: bool


class VerifierCase(BaseModel):
    """A fixed answer/evidence pair with known per-claim labels, for measuring the verifier alone."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: str
    question: str
    evidence: list[ChallengeEvidence] = Field(min_length=1)
    claims: list[ChallengeClaim] = Field(min_length=1)
    # correct: every claim true and supported by its citation
    # incorrect: at least one claim false or only partly true
    # correct_fact_wrong_citation: every claim true, but some citation does not support it
    answer_label: Literal["correct", "incorrect", "correct_fact_wrong_citation"]
    required: bool = False
    review_status: ReviewStatus = "proposed"
    notes: str = ""

    @property
    def answer(self) -> str:
        return " ".join(c.text for c in self.claims)


class TabularFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    operator: str
    value: str | float | int


SecurityCheck = Literal["no_code_execution", "no_shell_commands", "no_file_deletion", "no_filesystem_changes"]
Outcome = Literal["safe_refusal", "allowlisted_query", "expected_value"]


class TabularCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["count_vs_sum", "aggregate", "grouped", "row_count", "unsafe_input"]
    question: str
    expected_operation: Optional[str] = None
    expected_column: Optional[str] = None
    expected_filters: list[TabularFilter] = Field(default_factory=list)
    expected_group_by: Optional[str] = None
    expected_value: Optional[str] = None     # as rendered in the answer, e.g. "2650"
    # Outcomes that count as a pass; for ordinary cases only the expected value.
    acceptable_outcomes: list[Outcome] = Field(default_factory=lambda: ["expected_value"])
    # Checked through observable state (sandboxed files, patched execution
    # boundaries), not by reading the response text.
    security_checks: list[SecurityCheck] = Field(
        default_factory=lambda: ["no_code_execution", "no_shell_commands", "no_file_deletion", "no_filesystem_changes"]
    )
    review_status: ReviewStatus = "proposed"
    notes: str = ""


def _load(path: Path, model):
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    return _load(path, Case)


def load_verifier_cases(path: Path = VERIFIER_FILE) -> list[VerifierCase]:
    return _load(path, VerifierCase)


def load_tabular_cases(path: Path = TABULAR_FILE) -> list[TabularCase]:
    return _load(path, TabularCase)


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and map typographic quotes/dashes to ASCII.

    PDF extraction can turn straight apostrophes into curly ones, so labels and
    answers are compared in this normalized form.
    """
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


@lru_cache(maxsize=None)
def document_text(source: str, corpus_dir: str = str(CORPUS_DIR)) -> str:
    """Normalized text of a corpus document as RAGBase's ingestion extracts it."""
    import ingest

    entries = ingest.load_doc(str(Path(corpus_dir) / source))
    return normalize(" ".join(text for text, _, _ in entries))


def corpus_manifest(corpus_dir: Path = CORPUS_DIR) -> dict:
    return json.loads((corpus_dir / "MANIFEST.json").read_text())


def manifest_problems(corpus_dir: Path = CORPUS_DIR, tabular_dir: Path = TABULAR_DIR) -> list[str]:
    manifest = corpus_manifest(corpus_dir)
    problems = []
    for rel, digest in manifest["files"].items():
        folder, name = rel.split("/", 1)
        path = (corpus_dir if folder == corpus_dir.name else tabular_dir) / name
        if not path.exists():
            problems.append(f"manifest lists missing file {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"checksum mismatch for {rel}")
    return problems


def case_problems(cases: list[Case]) -> list[str]:
    """Consistency problems in the RAG cases (not correctness)."""
    problems = []
    ids = [c.id for c in cases]
    problems += [f"duplicate case id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for c in cases:
        if c.answerable == "yes" and not c.expected_facts:
            problems.append(f"{c.id}: answerable case has no expected facts")
        if c.answerable == "no" and c.expected_facts:
            problems.append(f"{c.id}: unanswerable case lists expected facts")
        if c.answerable in ("no", "partial") and not c.missing_information:
            problems.append(f"{c.id}: missing_information not described")
        for f in c.expected_facts:
            for ev in f.evidence:
                if not (CORPUS_DIR / ev.source).exists():
                    problems.append(f"{c.id}/{f.id}: unknown source {ev.source}")
                elif normalize(ev.quote) not in document_text(ev.source):
                    problems.append(f"{c.id}/{f.id}: quote not found in extracted text of {ev.source}")
            for group in f.keywords:
                if not any(normalize(k) in normalize(f.statement + " " + c.reference_answer) or
                           normalize(k) in " ".join(normalize(e.quote) for e in f.evidence) for k in group):
                    problems.append(f"{c.id}/{f.id}: keyword group {group} appears in neither statement nor evidence")
        for fb in c.forbidden_facts:
            try:
                re.compile(fb.pattern)
            except re.error as e:
                problems.append(f"{c.id}: invalid forbidden pattern ({e})")
    return problems


def indexed_chunks(corpus_dir: Path = CORPUS_DIR) -> dict[str, list[str]]:
    """Normalized chunk texts per source, produced by the real ingestion pipeline at default settings.

    Uses fake embeddings (chunking does not depend on them), so it runs without models.
    """
    import tempfile

    from langchain_core.embeddings import DeterministicFakeEmbedding

    import ingest
    import retriever

    with tempfile.TemporaryDirectory() as tmp:
        ingest.ingest_documents(ingest_dir=str(corpus_dir), index_dir=tmp, embed_model="fake",
                                embeddings=DeterministicFakeEmbedding(size=8))
        chunks: dict[str, list[str]] = {}
        for c in retriever.load_tfidf(tmp)[2]:
            chunks.setdefault(c["source"], []).append(normalize(c["text"]))
    return chunks


def chunk_problems(cases: list[Case], chunks: dict[str, list[str]]) -> list[str]:
    """Evidence quotes that do not sit inside a single indexed chunk (unmatchable in retrieval metrics)."""
    problems = []
    for c in cases:
        for f in c.expected_facts:
            for ev in f.evidence:
                if not any(normalize(ev.quote) in chunk for chunk in chunks.get(ev.source, [])):
                    problems.append(f"{c.id}/{f.id}: quote spans a chunk boundary in {ev.source}")
    return problems
