import os
import uuid

import asyncpg
import httpx
import pytest

from conftest import FakeProvider, final_response, no_sleep, tool_response
from factcheck.db import Database
from factcheck.api import app
from factcheck.data import load_answers
from factcheck.reliability import ProviderError, ProviderGate
from factcheck.spans import split_sentences
from factcheck.tracing import NullTracer

A01 = next(a for a in load_answers() if a.answer_id == "A01")
BODY = {"answer_id": "A01", "brand_id": "cleartrail", "question": A01.question, "ai_answer": A01.ai_answer,
        "agent_version": "v1"}
TEST_SCHEMA = "factcheck_test"


TEST_DB = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
EVIDENCE_TABLES = ("runs", "answer_results", "claim_results")


def client(provider, settings, store=None, timeout_s=1.0):
    st = app.state
    st.settings, st.provider, st.tracer, st.store = settings, provider, NullTracer(), store
    st.gate = ProviderGate(max_concurrent=3, timeout_s=timeout_s, retry_delay_s=0, sleep=no_sleep)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def ok_script():
    return [tool_response("cleartrail"),
            final_response([{"sentence_id": 4, "label": "supported", "fact_ids": ["cleartrail.mini.price_2026"],
                             "reason": "Prices match."}])]


async def test_classify_200(settings):
    async with client(FakeProvider(ok_script()), settings) as c:
        r = await c.post("/classify", json=BODY)
    assert r.status_code == 200
    claim = r.json()["claims"][0]
    assert claim["quote"] == A01.ai_answer[claim["start_char"]:claim["end_char"]]


@pytest.mark.parametrize("extra", [{"claims": []}, {"expected_label": "supported"}, {"gold_spans": []},
                                   {"agent_version": "v3"}, {"ai_answer": ""}])
async def test_classify_422(extra, settings):
    async with client(FakeProvider([]), settings) as c:
        r = await c.post("/classify", json={**BODY, **extra})
    assert r.status_code == 422


async def test_classify_404_unknown_brand(settings):
    async with client(FakeProvider([]), settings) as c:
        r = await c.post("/classify", json={**BODY, "brand_id": "rivalsound"})
    assert r.status_code == 404 and r.json()["error"]["type"] == "unknown_brand"


@pytest.mark.parametrize("script, status, etype", [
    ([ProviderError("auth", "provider authentication failed (401)", retryable=False)], 502, "provider_auth"),
    ([tool_response("cleartrail"), final_response([{"sentence_id": 1, "label": "maybe", "fact_ids": [], "reason": "x."}])],
     502, "invalid_output"),
])
async def test_classify_502(script, status, etype, settings):
    async with client(FakeProvider(script), settings) as c:
        r = await c.post("/classify", json=BODY)
    assert r.status_code == status and r.json()["error"]["type"] == etype
    assert "Traceback" not in r.text


async def test_classify_504_timeout(settings):
    async with client(FakeProvider(lambda req, n: tool_response("cleartrail"), delay=0.3), settings, timeout_s=0.02) as c:
        r = await c.post("/classify", json=BODY)
    assert r.status_code == 504 and r.json()["error"]["type"] == "provider_timeout"


async def test_evaluation_run_request_validation(settings):
    async with client(FakeProvider([]), settings) as c:
        assert (await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "final"})).status_code == 422
        assert (await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "check",
                                                         "labels": {}})).status_code == 422


needs_db = pytest.mark.skipif(not TEST_DB, reason="DATABASE_URL / TEST_DATABASE_URL not set")


async def _public_counts(store: Database) -> dict[str, int]:
    async with store.pool.acquire() as conn:
        exists = await conn.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename='runs'")
        if not exists:
            return {}
        return {t: await conn.fetchval(f"SELECT count(*) FROM public.{t}") for t in EVIDENCE_TABLES}


@pytest.fixture
async def store():
    s = await Database.connect(TEST_DB, schema=TEST_SCHEMA)
    async with s.pool.acquire() as conn:
        # Fully qualified so a reset can never reach the public evidence tables.
        await conn.execute(f"DROP TABLE IF EXISTS {TEST_SCHEMA}.claim_results, {TEST_SCHEMA}.answer_results, "
                           f"{TEST_SCHEMA}.runs CASCADE")
    await s.ensure_schema()
    before = await _public_counts(s)
    yield s
    assert await _public_counts(s) == before, "a test wrote to the public evidence tables"
    await s.close()


@needs_db
async def test_db_persistence_and_joined_retrieval(settings, store, tmp_path, monkeypatch):
    # Development split, three answers: one with claims, one valid empty, one provider error.
    import factcheck.runner as runner
    chosen = [a for a in load_answers() if a.answer_id in ("A01", "A02", "A03")]
    monkeypatch.setattr(runner, "answers_for_split", lambda split: chosen)

    def script(req, n):
        text = req.messages[0]["text"]
        if split_sentences(chosen[2].ai_answer)[3].text in text and req.mode == "tools":
            return ProviderError("auth", "denied", retryable=False)
        if req.mode == "tools":
            return tool_response("cleartrail", "cloudnote")
        if split_sentences(chosen[0].ai_answer)[3].text in text:
            return final_response([
                {"sentence_id": 5, "label": "contradicted", "fact_ids": ["cleartrail.mini.warranty_india"], "reason": "It is 12 months."},
                {"sentence_id": 4, "label": "supported", "fact_ids": ["cleartrail.mini.price_2026"], "reason": "Prices match."}])
        return final_response([])

    async with client(FakeProvider(script), settings, store=store) as c:
        r = await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "development"})
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "completed" and body["counts"]["error_answers"] == 1
        assert body["metrics"]["counts"]["M"] == 2 and body["metrics"]["counts"]["C"] == 2
        got = await c.get(f"/evaluation-runs/{body['run_id']}")
        assert got.status_code == 200
        run = got.json()
        assert [a["answer_id"] for a in run["answers"]] == ["A01", "A02", "A03"]
        a01, a02, a03 = run["answers"]
        assert [x["start_char"] for x in a01["claims"]] == sorted(x["start_char"] for x in a01["claims"])
        assert all(x["matched_gold"] and x["label_correct"] for x in a01["claims"])
        assert a02["status"] == "ok" and a02["claims"] == [] and a02["claim_count"] == 0
        assert a03["status"] == "error" and a03["error_type"] == "provider_auth" and a03["claims"] == []
        assert run["file_hashes"]["answers.csv"].startswith("fa28e8")
        assert (await c.get(f"/evaluation-runs/{uuid.uuid4()}")).status_code == 404
        assert (await c.get("/evaluation-runs/not-a-uuid")).status_code == 422

    run_id = uuid.UUID(body["run_id"])
    async with store.pool.acquire() as conn:
        rows = await conn.fetchval(f"SELECT count(*) FROM {TEST_SCHEMA}.answer_results WHERE run_id=$1", run_id)
        assert rows == 3
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(f"INSERT INTO {TEST_SCHEMA}.answer_results (run_id, answer_id, status, claim_count) VALUES ($1,'A02','ok',0)", run_id)
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(f"""INSERT INTO {TEST_SCHEMA}.claim_results (run_id, answer_id, claim_id, start_char,
                                  end_char, quote, label, fact_ids, reason) SELECT run_id, answer_id, claim_id || 'x',
                                  start_char, end_char, quote, label, fact_ids, reason FROM {TEST_SCHEMA}.claim_results
                                  WHERE run_id=$1 LIMIT 1""", run_id)
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute(f"""INSERT INTO {TEST_SCHEMA}.claim_results (run_id, answer_id, claim_id, start_char,
                                  end_char, quote, label, fact_ids, reason)
                                  VALUES ($1,'A99','A99:0:1',0,1,'x','unknown','{{}}','r.')""", run_id)


@needs_db
async def test_db_check_run_keeps_quality_null(settings, store):
    script = lambda req, n: tool_response("cleartrail", "cloudnote", "stonewave") if req.mode == "tools" else final_response([])
    async with client(FakeProvider(script), settings, store=store) as c:
        r = await c.post("/evaluation-runs", json={"agent_version": "v2", "split": "check"})
        assert r.status_code == 201 and r.json()["metrics"] is None
        run = (await c.get(f"/evaluation-runs/{r.json()['run_id']}")).json()
    assert run["metrics"] is None and len(run["answers"]) == 10
    assert all(a["scores"] is None for a in run["answers"])
