"""
citations.py — Evidence IDs and deterministic citation validation.

Shared by Standard RAG and Agentic RAG:

  1. Retrieved chunks become numbered evidence items (1, 2, ...), each
     bound to a real chunk_id, source file and location.
  2. The model is shown the evidence block and must cite only those numbers,
     in square brackets: [1], [2, 3].
  3. `validate_citations()` checks the answer without any model call:
     cited IDs must exist in the evidence that was actually provided;
     free-form citations like "[file.pdf @ p.3]" are flagged because the
     model could have invented them.

Citation identifiers therefore always originate from retrieval metadata.

Format note: citations are plain numbers, [1]. An earlier "E1" format matched
Presidio's US_DRIVER_LICENSE recognizer, so gateways that scan prompts for PII
(Prism) blocked every evidence prompt; "E-1", "#1" and other formats were
tagged as PERSON by spaCy in some contexts or lowered citation quality with a
small local model. Numbered citations, with the ID placed last in each
evidence tag, scanned clean and cited best. Model output in the older styles
([E1], [E-1], [#1]) is normalized to [1].
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from langchain_core.documents import Document

ABSTAIN_TEXT = "I don't have enough information in the provided documents to answer that."

_ABSTAIN_PATTERNS = [
    re.compile(r"\bdon'?t have enough information\b", re.I),
    re.compile(r"\bdo not have enough information\b", re.I),
    re.compile(r"\b(?:is|are) not (?:contained|mentioned|provided|stated) in the (?:provided )?(?:documents|context|evidence)\b", re.I),
    re.compile(r"\bthe (?:provided )?(?:documents|context|evidence) do(?:es)? not (?:contain|mention|state|specify)\b", re.I),
]

# Sentences that assert something is *absent* from the documents. They cannot
# be grounded by citing a passage, so claim verification skips them.
_ABSENCE_PATTERNS = _ABSTAIN_PATTERNS + [
    re.compile(r"\bnot (?:explicitly |specifically |directly )?(?:stated|mentioned|specified|provided|covered|addressed)\b", re.I),
    re.compile(r"\b(?:does|do) not (?:provide|include|say|cover|address|give)\b", re.I),
    re.compile(r"\bno (?:information|mention|details?)\b", re.I),
]

CITE_RE = re.compile(r"\[\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*\]")
_ID_RE = re.compile(r"\d{1,3}")
# Older / variant spellings a model may still produce: [E1], [E-1], [E 1], [#1].
_LEGACY_CITE_RE = re.compile(
    r"\[\s*((?:E[-\s]?|#)\d+(?:\s*[,;]\s*(?:E[-\s]?|#)?\d+)*)\s*\]"
)
_LEGACY_ID_RE = re.compile(r"(?:E[-\s]?|#)?(\d+)")
# Bracketed references that look like sources but are not evidence IDs,
# e.g. "[handbook.pdf @ p.2]" or "[Source: travel.txt]".
_NON_EVIDENCE_RE = re.compile(r"\[(?!\s*\d)[^\[\]\n]*(?:@|\.pdf|\.docx|\.txt|\.csv|\.xlsx|source)[^\[\]\n]*\]", re.I)
# Move citations that follow sentence punctuation onto the sentence they cite:
# "45 dollars. [1]" -> "45 dollars [1]."
_TRAILING_CITE_RE = re.compile(r"([.!?])\s*((?:\[\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*\]\s*)+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SKIP_LINE_RE = re.compile(r"^\s*(?:#+\s*|\*\*)?\s*(?:sources?|references?|citations?)\s*:?\**\s*", re.I)


@dataclass(frozen=True)
class Evidence:
    eid: str          # "1"
    chunk_id: str
    source: str
    loc: str
    text: str

    def label(self) -> str:
        return f"{self.source} @ {self.loc}" if self.loc else self.source


def _doc_chunk_id(doc: Document) -> str:
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    # Defensive fallback for documents built outside the ingest pipeline.
    return f"{doc.metadata.get('source', '')}|{doc.metadata.get('loc', '')}|{doc.page_content[:80]}"


def build_evidence(docs: Iterable[Document], start: int = 1) -> list[Evidence]:
    """Number retrieved documents as evidence, de-duplicated by chunk_id."""
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for doc in docs:
        cid = _doc_chunk_id(doc)
        if cid in seen:
            continue
        seen.add(cid)
        evidence.append(Evidence(
            eid=str(start + len(evidence)),
            chunk_id=cid,
            source=str(doc.metadata.get("source", "unknown")),
            loc=str(doc.metadata.get("loc", "")),
            text=doc.page_content,
        ))
    return evidence


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def format_evidence(evidence: list[Evidence]) -> str:
    """Render evidence for a prompt, one tagged block per item.

    Tag-style delimiters worked markedly better with small local models than
    a "[ID] source: ..." header, which they tended to copy verbatim instead
    of answering. The ID comes last in the tag: `<doc id="#29"` was tagged as
    a PERSON by spaCy-based PII scanners. Document text cannot close its own
    tag early.
    """
    if not evidence:
        return "(no evidence retrieved)"
    return "\n\n".join(
        f'<doc source="{_attr(e.source)}" location="{_attr(e.loc or "n/a")}" id="{e.eid}">\n'
        f"{re.sub(r'</?doc', '[doc', e.text.strip(), flags=re.I)}\n</doc>"
        for e in evidence
    )


def normalize_evidence_id(raw: str) -> str:
    """Map "E1", "E-1", "e1", "#1", "[1]" or "1" to the canonical "1"; other strings unchanged."""
    m = re.fullmatch(r"\s*\[?\s*(?:[Ee][-\s]?|#)?(\d+)\s*\]?\s*", str(raw))
    return str(int(m.group(1))) if m else str(raw)


def normalize_citations(text: str) -> tuple[str, int]:
    """Rewrite citation markers in older styles ([E1], [E-1], [#1]) to [1]. Returns (text, count)."""
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        count += 1
        ids = [str(int(n)) for n in _LEGACY_ID_RE.findall(m.group(1))]
        return "[" + ", ".join(ids) + "]"

    return _LEGACY_CITE_RE.sub(fix, text), count


def extract_citations(text: str) -> list[str]:
    """Ordered, unique evidence IDs cited in `text`."""
    ids: list[str] = []
    for m in CITE_RE.finditer(text):
        for eid in _ID_RE.findall(m.group(1)):
            if eid not in ids:
                ids.append(eid)
    return ids


def is_abstention(text: str) -> bool:
    return any(p.search(text) for p in _ABSTAIN_PATTERNS)


def is_absence_statement(claim: str) -> bool:
    """True for claims like "the evidence does not state X" (absence, not a fact to ground)."""
    return any(p.search(claim) for p in _ABSENCE_PATTERNS)


def split_claims(text: str) -> list[str]:
    """Split an answer into sentence-level claims, skipping headings and source lists."""
    normalized = _TRAILING_CITE_RE.sub(r" \2\1", text)
    claims = []
    for raw in _SENTENCE_SPLIT_RE.split(normalized):
        s = raw.strip().lstrip("-*•0123456789. )").strip()
        if not s or _SKIP_LINE_RE.match(s):
            continue
        words = CITE_RE.sub("", s).split()
        if len(words) < 4:  # fragments, list labels, bare citations
            continue
        claims.append(s)
    return claims


@dataclass
class CitationReport:
    cited_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    non_evidence_citations: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    total_claims: int = 0
    abstained: bool = False
    evidence_count: int = 0

    @property
    def citation_coverage(self) -> float:
        """Fraction of claims carrying at least one evidence citation."""
        if self.total_claims == 0:
            return 1.0
        return 1 - len(self.uncited_claims) / self.total_claims

    @property
    def ok(self) -> bool:
        """No fabricated references, and the answer is either cited or an abstention."""
        if self.invalid_ids or self.non_evidence_citations:
            return False
        return self.abstained or bool(self.valid_ids)

    def to_dict(self) -> dict:
        return {
            "cited_ids": self.cited_ids,
            "valid_ids": self.valid_ids,
            "invalid_ids": self.invalid_ids,
            "non_evidence_citations": self.non_evidence_citations,
            "uncited_claims": self.uncited_claims,
            "total_claims": self.total_claims,
            "citation_coverage": round(self.citation_coverage, 3),
            "abstained": self.abstained,
            "evidence_count": self.evidence_count,
            "ok": self.ok,
        }


def validate_citations(answer: str, evidence: list[Evidence]) -> CitationReport:
    """Deterministically check that every citation refers to provided evidence."""
    known = {e.eid for e in evidence}
    cited = extract_citations(answer)
    claims = split_claims(answer)
    # An answer that cites evidence is a (possibly partial) answer, not an
    # abstention, even if one of its sentences says something is missing.
    abstained = is_abstention(answer) and not cited
    return CitationReport(
        cited_ids=cited,
        valid_ids=[c for c in cited if c in known],
        invalid_ids=[c for c in cited if c not in known],
        non_evidence_citations=_NON_EVIDENCE_RE.findall(answer),
        uncited_claims=[] if abstained else [c for c in claims if not CITE_RE.search(c)],
        total_claims=len(claims),
        abstained=abstained,
        evidence_count=len(evidence),
    )


def resolve_citations(report: CitationReport, evidence: list[Evidence]) -> list[Evidence]:
    """Evidence items actually cited by the answer, in citation order."""
    by_id = {e.eid: e for e in evidence}
    return [by_id[c] for c in report.valid_ids]
