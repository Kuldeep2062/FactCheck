"""Sentence segmentation and validation of final claim spans.

The model never counts characters: it picks sentence numbers, and offsets come
from the regex match positions over the untouched ai_answer string.
"""
import re
from dataclasses import dataclass

from .config import LABELS
from .schemas import AgentOutput, Claim, ModelOutput

# A sentence runs from the first non-space character to terminal punctuation that is
# followed by whitespace or end of text, so decimals such as "1.5" stay inside it.
_SENTENCE = re.compile(r"\S.*?[.!?](?=\s|$)", re.S)
_REASON_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


class ValidationFailure(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.error_type = "invalid_output"


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    start_char: int
    end_char: int
    text: str


def split_sentences(text: str) -> list[Sentence]:
    return [Sentence(i, m.start(), m.end(), m.group()) for i, m in enumerate(_SENTENCE.finditer(text), start=1)]


def make_claim_id(answer_id: str, start: int, end: int) -> str:
    return f"{answer_id}:{start}:{end}"


def reason_sentence_count(reason: str) -> int:
    stripped = reason.strip()
    if not stripped:
        return 0
    ends = len(_REASON_SENTENCE_END.findall(stripped))
    return ends if stripped[-1] in ".!?" else ends + 1


def build_claims(answer_id: str, ai_answer: str, sentences: list[Sentence], output: ModelOutput,
                 allowed_fact_ids: set[str]) -> AgentOutput:
    by_id = {s.sentence_id: s for s in sentences}
    seen: set[int] = set()
    claims: list[Claim] = []
    for mc in output.claims:
        if mc.sentence_id not in by_id:
            raise ValidationFailure(f"sentence_id {mc.sentence_id} does not exist")
        if mc.sentence_id in seen:
            raise ValidationFailure(f"sentence_id {mc.sentence_id} returned more than once")
        seen.add(mc.sentence_id)
        bad = [f for f in mc.fact_ids if f not in allowed_fact_ids]
        if bad:
            raise ValidationFailure(f"fact_ids not returned by get_brand_facts: {bad}")
        s = by_id[mc.sentence_id]
        claims.append(Claim(
            claim_id=make_claim_id(answer_id, s.start_char, s.end_char),
            start_char=s.start_char, end_char=s.end_char, quote=ai_answer[s.start_char:s.end_char],
            label=mc.label, fact_ids=list(dict.fromkeys(mc.fact_ids)), reason=mc.reason.strip(),
        ))
    claims.sort(key=lambda c: c.start_char)
    result = AgentOutput(answer_id=answer_id, claims=claims)
    validate_output(result, ai_answer, allowed_fact_ids)
    return result


def validate_output(output: AgentOutput, ai_answer: str, valid_fact_ids: set[str] | frozenset[str]) -> None:
    """Contract check applied to every final output, whatever produced it."""
    ids: set[str] = set()
    prev_end = -1
    for c in output.claims:
        if not (0 <= c.start_char < c.end_char <= len(ai_answer)):
            raise ValidationFailure(f"span out of bounds: {c.claim_id}")
        if c.quote != ai_answer[c.start_char:c.end_char]:
            raise ValidationFailure(f"quote does not equal source slice: {c.claim_id}")
        if c.quote != c.quote.strip():
            raise ValidationFailure(f"quote has leading or trailing whitespace: {c.claim_id}")
        if c.claim_id != make_claim_id(output.answer_id, c.start_char, c.end_char):
            raise ValidationFailure(f"claim_id does not match span: {c.claim_id}")
        if c.claim_id in ids:
            raise ValidationFailure(f"duplicate claim_id {c.claim_id}")
        ids.add(c.claim_id)
        if c.start_char < prev_end:
            raise ValidationFailure(f"overlapping or unordered span: {c.claim_id}")
        prev_end = c.end_char
        if c.label not in LABELS:
            raise ValidationFailure(f"unsupported label {c.label!r}")
        unknown = [f for f in c.fact_ids if f not in valid_fact_ids]
        if unknown:
            raise ValidationFailure(f"unknown fact_ids {unknown}")
        if not 1 <= reason_sentence_count(c.reason) <= 2:
            raise ValidationFailure(f"reason must be one or two sentences: {c.claim_id}")
