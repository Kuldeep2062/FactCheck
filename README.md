# FactCheck: claim extraction and labelling agent

This agent reads a long AI answer about speakers and finds every factual claim sentence. It labels each one as **supported**, **contradicted** or **unknown**, using only the supplied catalogue, which it fetches through a live Gemini tool call. The repo also includes a deterministic evaluation harness, a FastAPI service with PostgreSQL storage, Langfuse tracing, and two frozen agent versions.

- `RESULTS.md` holds the measured v1 and v2 results, the trace-based diagnosis, limitations and disclosure.

## Requirements

- **Python 3.12.** 
- **uv 0.12** 
- **A PostgreSQL database.** Neon was used for the runs.
- **A Gemini API key**
- **Langfuse keys.**

Dependency versions are locked in `uv.lock`. 

## Setup

```bash
uv sync                     
cp .env.example .env        
```

```bash
uv run factcheck init-db
```

## Run

```bash
# the four required live runs, in order
uv run factcheck run --agent-version v1 --split development
uv run factcheck run --agent-version v2 --split development
uv run factcheck run --agent-version v1 --split check
uv run factcheck run --agent-version v2 --split check

uv run uvicorn factcheck.api:app --port 8000
curl -X POST localhost:8000/evaluation-runs -H 'content-type: application/json' \
     -d '{"agent_version":"v1","split":"check"}'
curl localhost:8000/evaluation-runs/<run_id>
```

## Test

```bash
uv run pytest -q
```

## Layout

```
data/                 issued CSVs, byte-identical (hashes asserted in tests/test_data.py)
src/factcheck/
  config.py           versions, settings
  data.py             CSV loading, typed fact values, file hashes
  facts_tool.py       get_brand_facts + call validation
  spans.py            sentence splitter, claim construction, output validation
  provider.py         provider-neutral turn + Gemini implementation, error classification
  reliability.py      ProviderGate: ≤3 in flight, 15 s timeout, one selective retry, RPM spacing
  agent.py            LangGraph agent (call_model → run_tools → call_model → finalize)
  prompts/v1.md, v2.md
  scoring.py          deterministic scorer
  runner.py           split runner, reports, Langfuse scores, DB writes
  db.py               PostgreSQL schema and queries (asyncpg, parameterized)
  api.py              FastAPI
  compare.py          v1 vs v2 comparison
  cli.py              `factcheck` command
tests/                pytest suite
runs/                 saved live-run reports and logs
```
