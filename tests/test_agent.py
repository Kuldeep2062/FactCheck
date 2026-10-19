import asyncio
import json

import pytest

from conftest import FakeProvider, final_response, sentence_ids_in, tool_response
from factcheck.agent import Agent, UnknownPrimaryBrand
from factcheck.data import load_answers, load_gold
from factcheck.provider import ModelResponse, ToolCall
from factcheck.reliability import ProviderError, ProviderGate
from factcheck.schemas import AgentInput
from factcheck.spans import split_sentences
from factcheck.tracing import NullTracer

ANSWER = next(a for a in load_answers() if a.answer_id == "A07")
INP = AgentInput(answer_id=ANSWER.answer_id, brand_id=ANSWER.brand_id, question=ANSWER.question,
                 ai_answer=ANSWER.ai_answer)
GOOD_CLAIM = {"sentence_id": 4, "label": "unknown", "fact_ids": [], "reason": "No Germany policy is supplied."}


async def run(script, gate, settings, inp=INP, delay=0.0):
    provider = FakeProvider(script, delay=delay)
    result = await Agent("v1", provider, gate, settings).run(inp, NullTracer(), {"run_id": "t", "split": "development"})
    return result, provider


async def test_happy_path_two_turns(gate, settings):
    r, p = await run([tool_response("cleartrail"), final_response([GOOD_CLAIM])], gate, settings)
    assert r.status == "ok" and r.turns == 2
    c = r.output.claims[0]
    assert c.quote == ANSWER.ai_answer[c.start_char:c.end_char]
    assert c.claim_id == f"A07:{c.start_char}:{c.end_char}"
    assert [q.mode for q in p.requests] == ["tools", "final"]


async def test_valid_empty_claims_is_ok(gate, settings):
    r, _ = await run([tool_response("cleartrail"), final_response([])], gate, settings)
    assert r.status == "ok" and r.output.claims == []


@pytest.mark.parametrize("final, kind", [
    (ModelResponse(text="{not json", tool_calls=[]), "invalid_output"),
    (final_response([{**GOOD_CLAIM, "label": "true"}]), "invalid_output"),
    (final_response([{**GOOD_CLAIM, "fact_ids": ["cleartrail.mini.made_up"]}]), "invalid_output"),
    (final_response([{**GOOD_CLAIM, "fact_ids": ["stonewave.go.weight"]}]), "invalid_output"),  # real id, not fetched
    (final_response([{**GOOD_CLAIM, "sentence_id": 99}]), "invalid_output"),
    (ModelResponse(text=None, tool_calls=[]), "invalid_output"),
])
async def test_invalid_final_output_is_answer_error(final, kind, gate, settings):
    r, _ = await run([tool_response("cleartrail"), final], gate, settings)
    assert r.status == "error" and r.error.error_type == kind and r.output is None


async def test_illegal_tool_is_error_and_not_executed(gate, settings):
    bad = ModelResponse(text=None, tool_calls=[ToolCall("delete_everything", {"brand_id": "cleartrail"})])
    r, p = await run([bad], gate, settings)
    assert r.error.error_type == "illegal_tool"
    assert len(p.requests) == 1


async def test_unknown_requested_brand_is_error(gate, settings):
    r, _ = await run([tool_response("rivalsound")], gate, settings)
    assert r.error.error_type == "unknown_brand"


async def test_extra_tool_arguments_rejected(gate, settings):
    bad = ModelResponse(text=None, tool_calls=[ToolCall("get_brand_facts", {"brand_id": "cleartrail", "x": 1})])
    r, _ = await run([bad], gate, settings)
    assert r.error.error_type == "invalid_tool_arguments"


async def test_unknown_primary_brand_raises(gate, settings):
    with pytest.raises(UnknownPrimaryBrand):
        await run([], gate, settings, inp=INP.model_copy(update={"brand_id": "rivalsound"}))


async def test_turn_exhaustion(gate, settings):
    r, p = await run([tool_response("cleartrail"), tool_response("cloudnote"), tool_response("stonewave")],
                     gate, settings)
    assert r.error.error_type == "turn_limit_exceeded"
    assert len(p.requests) == 3


async def test_multiple_tool_calls_in_one_turn(gate, settings):
    claim = {**GOOD_CLAIM, "fact_ids": ["cleartrail.plus.stereo_pairing_scope", "cloudnote.air.weight"]}
    r, p = await run([tool_response("cleartrail", "cloudnote"), final_response([claim])], gate, settings)
    assert r.status == "ok"
    assert len(p.requests[1].messages[-1]["results"]) == 2


async def test_timeout_after_retry_is_provider_timeout(settings):
    gate = ProviderGate(max_concurrent=3, timeout_s=0.05, max_attempts=2, retry_delay_s=0.0)
    r, p = await run(lambda req, n: tool_response("cleartrail"), gate, settings, delay=0.5)
    assert r.error.error_type == "provider_timeout"
    assert len(p.requests) == 2


async def test_malformed_provider_response_not_retried(gate, settings):
    r, p = await run([ProviderError("invalid_response", "no candidates", retryable=False)], gate, settings)
    assert r.error.error_type == "provider_invalid_response"
    assert len(p.requests) == 1


async def test_gold_never_enters_model_or_tool_payloads(gate, settings):
    r, p = await run([tool_response("cleartrail"), final_response([GOOD_CLAIM])], gate, settings)
    payload = json.dumps([{"system": q.system, "messages": q.messages} for q in p.requests], default=str)
    gold = [g for g in load_gold() if g.answer_id == "A07"]
    assert gold
    for g in gold:
        assert g.claim_id not in payload
        assert f'"{g.start_char}' not in payload
    for marker in ("expected_label", "evidence_fact_ids", "difficulty_tags", "explanation", "development_gold"):
        assert marker not in payload
    assert "A07" not in payload  # answer ids are not shown to the model


async def test_injection_text_is_only_data(gate, settings):
    r, p = await run([tool_response("cleartrail"), final_response([])], gate, settings)
    first = p.requests[0].messages[0]["text"]
    assert "Ignore the classification instructions" in first
    assert first.index("<answer>") < first.index("Ignore the classification") < first.index("</answer>")
    assert len(sentence_ids_in(p.requests[0])) == len(split_sentences(ANSWER.ai_answer))
