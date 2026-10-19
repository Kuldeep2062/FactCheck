import asyncio

import pytest

from conftest import FakeProvider, final_response, no_sleep, tool_response
from factcheck.data import answers_for_split
from factcheck.reliability import ProviderError, ProviderGate, RequestSpacer
from factcheck.runner import run_split
from factcheck.spans import split_sentences
from factcheck.tracing import NullTracer


async def test_gate_never_exceeds_three_active_requests():
    gate = ProviderGate(max_concurrent=3, timeout_s=2, sleep=no_sleep)
    active = peak = 0

    async def work(_attempt):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "ok"

    await asyncio.gather(*(gate.call(work) for _ in range(20)))
    assert peak == 3 and gate.peak_active == 3


@pytest.mark.parametrize("err", [
    ProviderError("rate_limit", "429", retryable=True, retry_after_s=30),
    ProviderError("transient", "503", retryable=True),
])
async def test_transient_errors_retried_once_then_succeed(err):
    delays = []

    async def rec_sleep(s):
        delays.append(s)

    gate = ProviderGate(timeout_s=1, retry_delay_s=2.0, retry_delay_cap_s=10.0, sleep=rec_sleep)
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        if attempt == 1:
            raise err
        return "ok"

    result, records = await gate.call(fn)
    assert result == "ok" and calls == [1, 2]
    assert [r.outcome for r in records] == ["error", "ok"]
    assert delays and delays[0] <= 10.0  # server hint of 30s is capped


async def test_at_most_two_attempts():
    gate = ProviderGate(timeout_s=1, sleep=no_sleep)
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise ProviderError("transient", "503", retryable=True)

    with pytest.raises(ProviderError) as ei:
        await gate.call(fn)
    assert calls == [1, 2] and len(ei.value.attempts) == 2


@pytest.mark.parametrize("kind", ["auth", "bad_request", "invalid_response"])
async def test_permanent_errors_not_retried(kind):
    gate = ProviderGate(timeout_s=1, sleep=no_sleep)
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise ProviderError(kind, kind, retryable=False)

    with pytest.raises(ProviderError):
        await gate.call(fn)
    assert calls == [1]


async def test_timeout_retried_once():
    gate = ProviderGate(timeout_s=0.02, sleep=no_sleep)
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        await asyncio.sleep(0.2)

    with pytest.raises(ProviderError) as ei:
        await gate.call(fn)
    assert ei.value.kind == "timeout" and calls == [1, 2]


async def test_request_spacer_spaces_starts():
    t = [0.0]
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    spacer = RequestSpacer(60, clock=lambda: t[0], sleep=fake_sleep)
    for _ in range(3):
        await spacer.wait()
    assert slept == [1.0, 2.0]


async def test_run_isolates_failures_orders_output_and_bounds_concurrency(settings, tmp_path):
    answers = answers_for_split("development")[:9]
    failing = {"A03", "A06"}

    def script(request, n):
        text = request.messages[0]["text"]
        is_fail = any(split_sentences(a.ai_answer)[3].text in text for a in answers if a.answer_id in failing)
        if request.mode == "tools":
            return ProviderError("auth", "denied", retryable=False) if is_fail else tool_response("cleartrail", "cloudnote", "stonewave")
        return final_response([])

    provider = FakeProvider(script, delay=0.01)
    gate = ProviderGate(max_concurrent=3, timeout_s=2, sleep=no_sleep)
    report = await run_split("v1", "development", provider=provider, settings=settings, tracer=NullTracer(),
                             answers=list(reversed(answers)), gate=gate, write_report=False)
    ids = [a["answer_id"] for a in report["answers"]]
    assert ids == sorted(ids) and len(ids) == 9
    errs = {a["answer_id"] for a in report["answers"] if a["status"] == "error"}
    assert errs == failing
    assert report["counts"]["error_answers"] == 2 and report["counts"]["empty_outputs"] == 7
    assert provider.peak <= 3 and provider.peak > 1  # overlapped, but bounded
    assert report["metrics"]["counts"]["G"] == sum(1 for _ in __import__("factcheck.data", fromlist=["x"]).load_gold()
                                                    if _.answer_id in {a.answer_id for a in answers})


async def test_check_run_has_null_quality(settings):
    provider = FakeProvider(lambda req, n: tool_response("cleartrail", "cloudnote", "stonewave")
                            if req.mode == "tools" else final_response([]))
    gate = ProviderGate(max_concurrent=3, timeout_s=2, sleep=no_sleep)
    report = await run_split("v1", "check", provider=provider, settings=settings, tracer=NullTracer(),
                             gate=gate, write_report=False)
    assert report["metrics"] is None and report["development_records"] is None
    assert report["counts"]["G"] is None and report["counts"]["M"] is None and report["counts"]["C"] is None
    assert len(report["answers"]) == 10 and all(a["scores"] is None for a in report["answers"])


async def test_token_budget_waits_for_refill():
    from factcheck.reliability import TokenBudget
    t = [0.0]
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        t[0] += s

    b = TokenBudget(6000, clock=lambda: t[0], sleep=fake_sleep)  # 100 tokens/second
    await b.acquire(5000)
    assert slept == []
    await b.acquire(3000)  # 1000 left, needs 2000 more -> 20 s
    assert slept == [pytest.approx(20.0)]
    await b.acquire(10_000)  # larger than capacity is clamped to capacity
    assert sum(slept) == pytest.approx(80.0)


async def test_gate_budget_wait_does_not_count_toward_timeout():
    t = [0.0]

    async def fake_sleep(s):
        t[0] += s

    gate = ProviderGate(timeout_s=0.5, tokens_per_minute=600, sleep=fake_sleep)
    gate.budget._clock = lambda: t[0]

    async def fn(_):
        return "ok"

    await gate.call(fn, estimated_tokens=600)
    result, _ = await gate.call(fn, estimated_tokens=600)  # waits 60 simulated seconds, then succeeds
    assert result == "ok" and gate.budget.waited_s == pytest.approx(60.0)


async def test_subset_run_is_labelled_and_scored_on_subset_gold(settings):
    provider = FakeProvider(lambda req, n: tool_response("cleartrail", "cloudnote", "stonewave")
                            if req.mode == "tools" else final_response([]))
    gate = ProviderGate(max_concurrent=3, timeout_s=2, sleep=no_sleep)
    report = await run_split("v1", "development", provider=provider, settings=settings, tracer=NullTracer(),
                             answer_ids=["A02", "A01"], gate=gate, write_report=False)
    assert report["settings"]["answer_subset"] == ["A01", "A02"]
    assert [a["answer_id"] for a in report["answers"]] == ["A01", "A02"]
    from factcheck.data import load_gold
    assert report["counts"]["G"] == sum(1 for g in load_gold() if g.answer_id in ("A01", "A02"))
    with pytest.raises(ValueError):
        await run_split("v1", "development", provider=provider, settings=settings, tracer=NullTracer(),
                        answer_ids=["A41"], gate=gate, write_report=False)


class _BrokenStore:
    def __init__(self):
        self.finished = []

    async def insert_run(self, meta):
        pass

    async def save_answers(self, run_id, answers):
        raise ConnectionError("connection lost")

    async def finish_run(self, run_id, status, counts, metrics, error=None):
        self.finished.append(status)


async def test_report_survives_database_failure(settings, tmp_path):
    import json
    provider = FakeProvider(lambda req, n: tool_response("cleartrail", "cloudnote", "stonewave")
                            if req.mode == "tools" else final_response([]))
    gate = ProviderGate(max_concurrent=3, timeout_s=2, sleep=no_sleep)
    store = _BrokenStore()
    report = await run_split("v1", "development", provider=provider, settings=settings, tracer=NullTracer(),
                             answer_ids=["A01", "A02"], gate=gate, store=store, runs_dir=tmp_path)
    assert report["status"] == "failed" and store.finished == ["failed"]
    assert report["persistence_errors"][0].startswith("database: ConnectionError")
    saved = json.loads((tmp_path / report["report_path"].split("/")[-1] / "report.json").read_text())
    assert saved["status"] == "failed" and len(saved["answers"]) == 2 and saved["metrics"] is not None
