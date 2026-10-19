"""PostgreSQL persistence with asyncpg. Values are always $n parameters; the schema name is a validated identifier."""
import json
import os
import re
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {s}.runs (
    run_id            uuid PRIMARY KEY,
    split             text NOT NULL CHECK (split IN ('development', 'check')),
    agent_version     text NOT NULL CHECK (agent_version IN ('v1', 'v2')),
    status            text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    prompt_version    text NOT NULL,
    dataset_version   text NOT NULL,
    evaluator_version text NOT NULL,
    fact_version      text NOT NULL,
    provider          text NOT NULL,
    model             text NOT NULL,
    settings          jsonb NOT NULL,
    file_hashes       jsonb NOT NULL,
    source_revision   text,
    counts            jsonb,
    metrics           jsonb,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);

CREATE TABLE IF NOT EXISTS {s}.answer_results (
    run_id            uuid NOT NULL REFERENCES {s}.runs(run_id) ON DELETE CASCADE,
    answer_id         text NOT NULL,
    status            text NOT NULL CHECK (status IN ('ok', 'error')),
    error_type        text,
    error_message     text,
    claim_count       integer NOT NULL CHECK (claim_count >= 0),
    latency_ms        integer,
    model_turns       integer,
    provider_attempts integer,
    usage             jsonb,
    trace_id          text,
    scores            jsonb,
    PRIMARY KEY (run_id, answer_id),
    CHECK ((status = 'error') = (error_type IS NOT NULL)),
    CHECK (status = 'ok' OR claim_count = 0)
);

CREATE TABLE IF NOT EXISTS {s}.claim_results (
    run_id         uuid NOT NULL,
    answer_id      text NOT NULL,
    claim_id       text NOT NULL,
    start_char     integer NOT NULL CHECK (start_char >= 0),
    end_char       integer NOT NULL,
    quote          text NOT NULL,
    label          text NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
    fact_ids       text[] NOT NULL,
    reason         text NOT NULL,
    matched_gold   boolean,
    expected_label text,
    label_correct  boolean,
    PRIMARY KEY (run_id, answer_id, start_char, end_char),
    UNIQUE (run_id, claim_id),
    CHECK (end_char > start_char),
    FOREIGN KEY (run_id, answer_id) REFERENCES {s}.answer_results(run_id, answer_id) ON DELETE CASCADE
);
"""


async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


def normalize_dsn(dsn: str) -> str:
    # asyncpg rejects libpq's channel_binding option; sslmode=require still enforces TLS.
    parts = urlsplit(dsn)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "channel_binding"]
    return urlunsplit(parts._replace(query=urlencode(query)))


class Database:
    """Every statement names its schema explicitly.

    Session-level search_path is not safe here: asyncpg runs RESET ALL when a connection
    returns to the pool, which silently sends later queries back to public.
    """

    def __init__(self, pool: asyncpg.Pool, schema: str = "public"):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", schema):
            raise ValueError(f"invalid schema name {schema!r}")
        self.pool = pool
        self.schema = schema

    def _q(self, sql: str) -> str:
        return sql.replace("{s}", self.schema)

    @classmethod
    async def connect(cls, dsn: str | None = None, schema: str = "public") -> "Database":
        dsn = normalize_dsn(dsn or os.environ["DATABASE_URL"])
        # Neon's -pooler endpoint is PgBouncer; a disabled statement cache avoids prepared-statement clashes.
        # command_timeout turns a connection killed by a network change into an error instead of a hang.
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, init=_init_conn, statement_cache_size=0,
                                         timeout=20, command_timeout=30)
        return cls(pool, schema)

    async def close(self) -> None:
        await self.pool.close()

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(self._q("CREATE SCHEMA IF NOT EXISTS {s};" + SCHEMA_SQL))

    async def insert_run(self, run: dict[str, Any]) -> None:
        await self.pool.execute(self._q(
            """INSERT INTO {s}.runs (run_id, split, agent_version, status, prompt_version, dataset_version,
                   evaluator_version, fact_version, provider, model, settings, file_hashes, source_revision)
               VALUES ($1,$2,$3,'running',$4,$5,$6,$7,$8,$9,$10,$11,$12)"""),
            run["run_id"], run["split"], run["agent_version"], run["prompt_version"], run["dataset_version"],
            run["evaluator_version"], run["fact_version"], run["provider"], run["model"], run["settings"],
            run["file_hashes"], run.get("source_revision"),
        )

    async def save_answers(self, run_id: uuid.UUID, answers: list[dict]) -> None:
        """One transaction for all answer and claim rows of a run."""
        async with self.pool.acquire() as conn, conn.transaction():
            for a in answers:
                await conn.execute(self._q(
                    """INSERT INTO {s}.answer_results (run_id, answer_id, status, error_type, error_message,
                           claim_count, latency_ms, model_turns, provider_attempts, usage, trace_id, scores)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)"""),
                    run_id, a["answer_id"], a["status"], a.get("error_type"), a.get("error_message"),
                    len(a["claims"]), a.get("latency_ms"), a.get("model_turns"), a.get("provider_attempts"),
                    a.get("usage"), a.get("trace_id"), a.get("scores"),
                )
                if a["claims"]:
                    await conn.executemany(self._q(
                        """INSERT INTO {s}.claim_results (run_id, answer_id, claim_id, start_char, end_char, quote,
                               label, fact_ids, reason, matched_gold, expected_label, label_correct)
                           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)"""),
                        [(run_id, a["answer_id"], c["claim_id"], c["start_char"], c["end_char"], c["quote"],
                          c["label"], c["fact_ids"], c["reason"], c.get("matched_gold"), c.get("expected_label"),
                          c.get("label_correct")) for c in a["claims"]],
                    )

    async def finish_run(self, run_id: uuid.UUID, status: str, counts: dict | None, metrics: dict | None,
                         error: str | None = None) -> None:
        await self.pool.execute(self._q(
            "UPDATE {s}.runs SET status=$2, counts=$3, metrics=$4, error=$5, finished_at=now() WHERE run_id=$1"),
            run_id, status, counts, metrics, error,
        )

    async def get_run(self, run_id: uuid.UUID) -> dict | None:
        async with self.pool.acquire() as conn:
            run = await conn.fetchrow(self._q("SELECT * FROM {s}.runs WHERE run_id=$1"), run_id)
            if run is None:
                return None
            rows = await conn.fetch(self._q(
                """SELECT a.answer_id, a.status, a.error_type, a.error_message, a.claim_count, a.latency_ms,
                          a.model_turns, a.provider_attempts, a.usage, a.trace_id, a.scores,
                          c.claim_id, c.start_char, c.end_char, c.quote, c.label, c.fact_ids, c.reason,
                          c.matched_gold, c.expected_label, c.label_correct
                   FROM {s}.answer_results a
                   LEFT JOIN {s}.claim_results c ON c.run_id = a.run_id AND c.answer_id = a.answer_id
                   WHERE a.run_id = $1
                   ORDER BY a.answer_id, c.start_char NULLS FIRST, c.end_char"""),
                run_id,
            )
        answers: dict[str, dict] = {}
        claim_cols = ("claim_id", "start_char", "end_char", "quote", "label", "fact_ids", "reason",
                      "matched_gold", "expected_label", "label_correct")
        for r in rows:
            a = answers.get(r["answer_id"])
            if a is None:
                a = {k: r[k] for k in r.keys() if k not in claim_cols}
                a["claims"] = []
                answers[r["answer_id"]] = a
            if r["claim_id"] is not None:
                claim = {k: r[k] for k in claim_cols}
                claim["fact_ids"] = list(claim["fact_ids"])
                a["claims"].append(claim)
        out = dict(run)
        for k in ("created_at", "finished_at"):
            out[k] = out[k].isoformat() if out[k] else None
        out["run_id"] = str(out["run_id"])
        out["answers"] = list(answers.values())
        return out
