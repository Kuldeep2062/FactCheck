import pytest

from factcheck.data import load_answers, load_gold
from factcheck.schemas import AgentOutput, Claim, ModelClaim, ModelOutput
from factcheck.spans import ValidationFailure, build_claims, reason_sentence_count, split_sentences, validate_output

TEXT = "Intro advice here. The Mini costs INR 1,299. It weighs 0.32 kg."
FACTS = {"cleartrail.mini.price_2026", "cleartrail.mini.weight"}


def claim(start, end, label="supported", answer_id="X", text=TEXT, **kw):
    base = dict(claim_id=f"{answer_id}:{start}:{end}", start_char=start, end_char=end, quote=text[start:end],
                label=label, fact_ids=[], reason="Matches the catalogue.")
    base.update(kw)
    return Claim.model_construct(**base)


def test_splitter_reproduces_every_gold_span():
    texts = {a.answer_id: a.ai_answer for a in load_answers()}
    spans = {a: {(s.start_char, s.end_char) for s in split_sentences(t)} for a, t in texts.items()}
    assert all((g.start_char, g.end_char) in spans[g.answer_id] for g in load_gold())


def test_splitter_keeps_decimals_inside_sentence():
    s = split_sentences(TEXT)
    assert [x.text for x in s] == ["Intro advice here.", "The Mini costs INR 1,299.", "It weighs 0.32 kg."]


def test_valid_output_passes():
    s = split_sentences(TEXT)
    validate_output(AgentOutput(answer_id="X", claims=[claim(s[1].start_char, s[1].end_char)]), TEXT, FACTS)


@pytest.mark.parametrize("bad", [
    lambda: claim(0, len(TEXT) + 5),
    lambda: claim(19, 44, quote="The Mini costs INR 999."),
    lambda: claim(18, 44),  # leading whitespace inside span
    lambda: claim(19, 44, claim_id="X:0:1"),
    lambda: claim(19, 44, label="true"),
    lambda: claim(19, 44, fact_ids=["made.up.fact"]),
    lambda: claim(19, 44, reason="One. Two. Three."),
])
def test_invalid_claims_rejected(bad):
    with pytest.raises(ValidationFailure):
        validate_output(AgentOutput.model_construct(answer_id="X", claims=[bad()]), TEXT, FACTS)


def test_duplicate_and_overlapping_spans_rejected():
    with pytest.raises(ValidationFailure, match="duplicate"):
        validate_output(AgentOutput.model_construct(answer_id="X", claims=[claim(19, 44), claim(19, 44)]), TEXT, FACTS)
    with pytest.raises(ValidationFailure, match="overlapping"):
        validate_output(AgentOutput.model_construct(answer_id="X", claims=[claim(19, 44), claim(30, 44)]), TEXT, FACTS)


def test_build_claims_derives_offsets_and_orders():
    s = split_sentences(TEXT)
    out = build_claims("X", TEXT, s, ModelOutput(claims=[
        ModelClaim(sentence_id=3, label="contradicted", fact_ids=["cleartrail.mini.weight"], reason="It is 320 g."),
        ModelClaim(sentence_id=2, label="supported", fact_ids=["cleartrail.mini.price_2026"], reason="Matches."),
    ]), FACTS)
    assert [c.claim_id for c in out.claims] == [f"X:{s[1].start_char}:{s[1].end_char}", f"X:{s[2].start_char}:{s[2].end_char}"]
    assert all(c.quote == TEXT[c.start_char:c.end_char] for c in out.claims)


@pytest.mark.parametrize("mc, msg", [
    (ModelClaim(sentence_id=9, label="supported", fact_ids=[], reason="x."), "does not exist"),
    (ModelClaim(sentence_id=2, label="supported", fact_ids=["cleartrail.plus.price_2026"], reason="x."), "fact_ids"),
])
def test_build_claims_rejects_bad_model_claims(mc, msg):
    with pytest.raises(ValidationFailure, match=msg):
        build_claims("X", TEXT, split_sentences(TEXT), ModelOutput(claims=[mc]), FACTS)


def test_build_claims_rejects_repeated_sentence():
    mc = ModelClaim(sentence_id=2, label="supported", fact_ids=[], reason="x.")
    with pytest.raises(ValidationFailure, match="more than once"):
        build_claims("X", TEXT, split_sentences(TEXT), ModelOutput(claims=[mc, mc]), FACTS)


def test_reason_sentence_count():
    assert reason_sentence_count("Warranty is 1.5 years.") == 1
    assert reason_sentence_count("A. B.") == 2
    assert reason_sentence_count("no punctuation") == 1
    assert reason_sentence_count("") == 0
