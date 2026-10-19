"""Deterministic scorer: exact (answer_id, start_char, end_char) matching, no fuzzy or LLM judging."""
from dataclasses import dataclass

from .config import LABELS
from .data import GoldClaim

EXPECTED_ROWS = (*LABELS, "no_gold")
PREDICTED_COLS = (*LABELS, "missing_or_error")


class ScoringError(Exception):
    pass


@dataclass(frozen=True)
class PredictedClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str


@dataclass(frozen=True)
class AnswerOutcome:
    answer_id: str
    status: str  # ok | error
    claims: tuple[PredictedClaim, ...] = ()


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _f1(hits: int, predicted: int, actual: int) -> float:
    return _ratio(2 * hits, predicted + actual)


def _prf(hits: int, predicted: int, actual: int) -> dict:
    return {"precision": _ratio(hits, predicted), "recall": _ratio(hits, actual), "f1": _f1(hits, predicted, actual)}


def score_split(scheduled_answer_ids: list[str], outcomes: list[AnswerOutcome], gold: list[GoldClaim],
                answer_texts: dict[str, str]) -> dict:
    scheduled = list(scheduled_answer_ids)
    if len(set(scheduled)) != len(scheduled):
        raise ScoringError("scheduled answer ids contain duplicates")
    seen: dict[str, AnswerOutcome] = {}
    for o in outcomes:
        if o.answer_id in seen:
            raise ScoringError(f"duplicate outcome for {o.answer_id}")
        if o.answer_id not in scheduled:
            raise ScoringError(f"outcome for unscheduled answer {o.answer_id}")
        seen[o.answer_id] = o
    missing = [a for a in scheduled if a not in seen]
    if missing:
        raise ScoringError(f"missing outcomes for {missing}")

    gold_in_split = [g for g in gold if g.answer_id in seen]
    gold_by_span = {(g.answer_id, g.start_char, g.end_char): g for g in gold_in_split}

    preds: list[PredictedClaim] = []
    for o in (seen[a] for a in scheduled):
        if o.status == "error":
            if o.claims:
                raise ScoringError(f"error answer {o.answer_id} carries predictions")
            continue
        for c in o.claims:
            if c.quote != answer_texts[c.answer_id][c.start_char:c.end_char]:
                raise ScoringError(f"prediction quote does not equal source slice: {c.claim_id}")
            preds.append(c)
    pred_keys = [(p.answer_id, p.start_char, p.end_char) for p in preds]
    if len(set(pred_keys)) != len(pred_keys):
        raise ScoringError("duplicate predicted spans")

    matched: list[tuple[PredictedClaim, GoldClaim]] = []
    unmatched_preds: list[PredictedClaim] = []
    matched_gold_ids: set[str] = set()
    for p in preds:
        g = gold_by_span.get((p.answer_id, p.start_char, p.end_char))
        if g is None:
            unmatched_preds.append(p)
        else:
            matched.append((p, g))
            matched_gold_ids.add(g.claim_id)

    G, P, M = len(gold_in_split), len(preds), len(matched)
    C = sum(1 for p, g in matched if p.label == g.expected_label)

    confusion = {r: {c: 0 for c in PREDICTED_COLS} for r in EXPECTED_ROWS}
    for p, g in matched:
        confusion[g.expected_label][p.label] += 1
    for g in gold_in_split:
        if g.claim_id not in matched_gold_ids:
            confusion[g.expected_label]["missing_or_error"] += 1
    for p in unmatched_preds:
        confusion["no_gold"][p.label] += 1

    per_label = {}
    for label in LABELS:
        tp = sum(1 for p, g in matched if p.label == label and g.expected_label == label)
        predicted = sum(1 for p in preds if p.label == label)
        actual = sum(1 for g in gold_in_split if g.expected_label == label)
        per_label[label] = {**_prf(tp, predicted, actual), "tp": tp, "predicted": predicted, "actual": actual}

    error_answers = [a for a in scheduled if seen[a].status == "error"]
    per_answer = {}
    for a in scheduled:
        g_a = [g for g in gold_in_split if g.answer_id == a]
        p_a = [p for p in preds if p.answer_id == a]
        m_a = [(p, g) for p, g in matched if p.answer_id == a]
        c_a = sum(1 for p, g in m_a if p.label == g.expected_label)
        per_answer[a] = {
            "status": seen[a].status, "gold": len(g_a), "predicted": len(p_a), "matched": len(m_a), "correct": c_a,
            "extraction_f1": _f1(len(m_a), len(p_a), len(g_a)) if (g_a or p_a) else None,
            "joint_f1": _f1(c_a, len(p_a), len(g_a)) if (g_a or p_a) else None,
            "missing_gold_ids": [g.claim_id for g in g_a if g.claim_id not in matched_gold_ids],
            "extra_claim_ids": [p.claim_id for p in unmatched_preds if p.answer_id == a],
            "wrong_label": [{"claim_id": g.claim_id, "expected": g.expected_label, "predicted": p.label}
                            for p, g in m_a if p.label != g.expected_label],
            "label_results": [{"claim_id": g.claim_id, "expected": g.expected_label, "predicted": p.label,
                               "correct": p.label == g.expected_label} for p, g in m_a],
        }

    return {
        "counts": {"answers": len(scheduled), "answer_errors": len(error_answers), "G": G, "P": P, "M": M, "C": C,
                   "unmatched_predictions": len(unmatched_preds)},
        "extraction": _prf(M, P, G),
        "joint": _prf(C, P, G),
        "gold_claim_coverage": _ratio(C, G),
        "per_label": per_label,
        "macro_f1": sum(per_label[l]["f1"] for l in LABELS) / len(LABELS),
        "confusion_matrix": confusion,
        "missed_gold_ids": sorted(g.claim_id for g in gold_in_split if g.claim_id not in matched_gold_ids),
        "unmatched_prediction_ids": sorted(p.claim_id for p in unmatched_preds),
        "wrong_label_ids": sorted(g.claim_id for p, g in matched if p.label != g.expected_label),
        "error_answer_ids": error_answers,
        "per_answer": per_answer,
    }
