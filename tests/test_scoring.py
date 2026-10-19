import pytest

from factcheck.data import GoldClaim
from factcheck.scoring import AnswerOutcome, PredictedClaim, ScoringError, score_split

# Ten one-word "sentences" per answer keep span arithmetic obvious.
TEXT = " ".join(f"s{i}." for i in range(12))
SPANS = [(i * 4, i * 4 + 3) for i in range(12)]
LABELS = ["supported", "contradicted", "unknown", "supported", "contradicted",
          "unknown", "supported", "contradicted", "unknown", "supported"]


def gold(n=10, answer="A1"):
    return [GoldClaim(answer, f"{answer}:{s}:{e}", s, e, TEXT[s:e], LABELS[i], ()) for i, (s, e) in enumerate(SPANS[:n])]


def pred(i, label, answer="A1"):
    s, e = SPANS[i]
    return PredictedClaim(answer, f"{answer}:{s}:{e}", s, e, TEXT[s:e], label)


def run(outcomes, scheduled=("A1",), g=None):
    return score_split(list(scheduled), outcomes, g if g is not None else gold(), {a: TEXT for a in scheduled})


def test_perfect_output():
    m = run([AnswerOutcome("A1", "ok", tuple(pred(i, LABELS[i]) for i in range(10)))])
    assert m["extraction"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert m["joint"]["f1"] == 1.0 and m["macro_f1"] == 1.0
    assert m["missed_gold_ids"] == [] and m["unmatched_prediction_ids"] == [] and m["wrong_label_ids"] == []


def test_assignment_example_g10_p10_m8_c7():
    preds = [pred(i, LABELS[i]) for i in range(7)]
    preds.append(pred(7, "supported"))          # matched, wrong label (gold contradicted)
    preds += [pred(10, "unknown"), pred(11, "supported")]  # two extra spans, no gold
    m = run([AnswerOutcome("A1", "ok", tuple(preds))])
    assert (m["counts"]["G"], m["counts"]["P"], m["counts"]["M"], m["counts"]["C"]) == (10, 10, 8, 7)
    for part, val in (("extraction", 0.8), ("joint", 0.7)):
        for k in ("precision", "recall", "f1"):
            assert m[part][k] == pytest.approx(val)
    assert m["gold_claim_coverage"] == pytest.approx(0.7)
    cm = m["confusion_matrix"]
    total = sum(sum(r.values()) for r in cm.values())
    assert total == 10 + 2
    assert cm["no_gold"]["missing_or_error"] == 0
    assert cm["contradicted"]["supported"] == 1
    assert cm["unknown"]["missing_or_error"] == 1 and cm["supported"]["missing_or_error"] == 1


def test_missed_claims_reduce_recall_only():
    m = run([AnswerOutcome("A1", "ok", tuple(pred(i, LABELS[i]) for i in range(6)))])
    assert m["extraction"]["precision"] == 1.0
    assert m["extraction"]["recall"] == pytest.approx(0.6)
    assert len(m["missed_gold_ids"]) == 4


def test_extra_advice_extracted_reduces_precision():
    preds = [pred(i, LABELS[i]) for i in range(10)] + [pred(10, "unknown")]
    m = run([AnswerOutcome("A1", "ok", tuple(preds))])
    assert m["extraction"]["precision"] == pytest.approx(10 / 11)
    assert m["extraction"]["recall"] == 1.0
    assert m["unmatched_prediction_ids"] == ["A1:40:43"]
    assert m["confusion_matrix"]["no_gold"]["unknown"] == 1
    assert m["per_label"]["unknown"]["predicted"] == 4  # unmatched span still counts as a predicted positive


def test_wrong_labels_affect_joint_and_per_label():
    preds = [pred(i, "unknown") for i in range(10)]
    m = run([AnswerOutcome("A1", "ok", tuple(preds))])
    assert m["extraction"]["f1"] == 1.0
    assert m["joint"]["f1"] == pytest.approx(0.3)
    assert m["per_label"]["supported"]["f1"] == 0.0
    assert m["per_label"]["unknown"]["precision"] == pytest.approx(0.3)
    assert m["per_label"]["unknown"]["recall"] == 1.0
    assert m["macro_f1"] == pytest.approx((0 + 0 + 2 * 3 / 13) / 3)


def test_provider_error_answer_keeps_gold_in_denominator():
    g = gold() + gold(answer="A2")
    m = score_split(["A1", "A2"], [AnswerOutcome("A1", "ok", tuple(pred(i, LABELS[i]) for i in range(10))),
                                   AnswerOutcome("A2", "error")], g, {"A1": TEXT, "A2": TEXT})
    assert m["counts"]["G"] == 20 and m["counts"]["answer_errors"] == 1
    assert m["joint"]["recall"] == pytest.approx(0.5)
    assert m["error_answer_ids"] == ["A2"]
    assert sum(1 for x in m["missed_gold_ids"] if x.startswith("A2:")) == 10


def test_valid_empty_predictions_score_zero():
    m = run([AnswerOutcome("A1", "ok", ())])
    assert m["counts"]["P"] == 0
    assert m["extraction"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert m["per_answer"]["A1"]["status"] == "ok"


def test_rejects_missing_or_duplicated_outcomes():
    with pytest.raises(ScoringError, match="missing"):
        score_split(["A1", "A2"], [AnswerOutcome("A1", "ok")], gold(), {"A1": TEXT, "A2": TEXT})
    with pytest.raises(ScoringError, match="duplicate"):
        run([AnswerOutcome("A1", "ok"), AnswerOutcome("A1", "ok")])


def test_rejects_error_answer_with_predictions_and_bad_quotes():
    with pytest.raises(ScoringError):
        run([AnswerOutcome("A1", "error", (pred(0, "supported"),))])
    bad = PredictedClaim("A1", "A1:0:3", 0, 3, "zz.", "supported")
    with pytest.raises(ScoringError, match="quote"):
        run([AnswerOutcome("A1", "ok", (bad,))])


def test_partial_span_does_not_match():
    partial = PredictedClaim("A1", "A1:0:2", 0, 2, TEXT[0:2], "supported")
    m = run([AnswerOutcome("A1", "ok", (partial,))])
    assert m["counts"]["M"] == 0 and m["unmatched_prediction_ids"] == ["A1:0:2"]
