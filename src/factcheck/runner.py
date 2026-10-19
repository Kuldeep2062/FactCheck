"""Runs one agent version over one fixed CSV split and records every outcome."""
import asyncio
import datetime as dt
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from .agent import Agent, AgentResult, UnknownPrimaryBrand
from . import config
from .config import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, ROOT, Settings
from .data import answers_for_split, data_file_hashes, load_gold
from .reliability import ProviderGate
from .schemas import AgentError, AgentInput
from .scoring import AnswerOutcome, PredictedClaim, score_split


def code_fingerprint() -> str:
    """sha256 over the agent package sources and prompts, so a run proves which frozen code produced it."""
    h = hashlib.sha256()
    pkg = ROOT / "src" / "factcheck"
    for f in sorted(list(pkg.glob("*.py")) + list((pkg / "prompts").glob("*.md"))):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def source_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if rev.returncode != 0:
            return f"tree-sha256:{code_fingerprint()} (no git commit yet)"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def make_gate(settings: Settings, sleep=asyncio.sleep) -> ProviderGate:
    return ProviderGate(
        max_concurrent=settings.max_concurrent_requests, timeout_s=settings.request_timeout_s,
        max_attempts=settings.max_attempts_per_turn, retry_delay_s=settings.retry_delay_s,
        retry_delay_cap_s=settings.retry_delay_cap_s, requests_per_minute=settings.requests_per_minute,
        tokens_per_minute=settings.tokens_per_minute, sleep=sleep,
    )


def _error_result(answer_id: str, error_type: str, message: str) -> AgentResult:
    return AgentResult(answer_id=answer_id, status="error", output=None,
                       error=AgentError(error_type=error_type, message=message), latency_ms=0, turns=0,
                       provider_attempts=0, usage={}, trace_id=None)


async def run_split(agent_version: str, split: str, *, provider, settings: Settings, tracer, store=None,
                    answers=None, answer_ids: list[str] | None = None, gate: ProviderGate | None = None,
                    write_report: bool = True, runs_dir: Path | None = None) -> dict:
    run_id = uuid.uuid4()
    pool_of_answers = answers if answers is not None else answers_for_split(split)
    if answer_ids is not None:
        by_id = {a.answer_id: a for a in pool_of_answers}
        unknown = [i for i in answer_ids if i not in by_id]
        if unknown or len(set(answer_ids)) != len(answer_ids):
            raise ValueError(f"answer_ids must be unique ids from the {split} split; bad: {unknown}")
        pool_of_answers = [by_id[i] for i in answer_ids]
    scheduled = sorted(pool_of_answers, key=lambda a: a.answer_id)
    full_split = {a.answer_id for a in answers_for_split(split)}
    subset = sorted(a.answer_id for a in scheduled) if {a.answer_id for a in scheduled} != full_split else None
    gate = gate or make_gate(settings)
    agent = Agent(agent_version, provider, gate, settings)
    started_at = dt.datetime.now(dt.timezone.utc)
    meta = {
        "run_id": run_id, "split": split, "agent_version": agent_version, "prompt_version": agent.prompt_version,
        "dataset_version": DATASET_VERSION, "evaluator_version": EVALUATOR_VERSION, "fact_version": FACT_VERSION,
        "provider": provider.name, "model": provider.model,
        # answer_subset is null for a full split; a subset run says so in its own settings.
        "settings": {**settings.as_dict(), "answer_subset": subset},
        "file_hashes": data_file_hashes(), "source_revision": source_revision(),
    }
    if store is not None:
        await store.insert_run(meta)

    trace_meta = {"run_id": str(run_id), "split": split, "dataset_version": DATASET_VERSION,
                  "fact_version": FACT_VERSION, "evaluator_version": EVALUATOR_VERSION}

    async def one(ans) -> AgentResult:
        inp = AgentInput(answer_id=ans.answer_id, brand_id=ans.brand_id, question=ans.question, ai_answer=ans.ai_answer)
        try:
            return await agent.run(inp, tracer, trace_meta)
        except UnknownPrimaryBrand:
            return _error_result(ans.answer_id, "unknown_brand", f"unknown primary brand {ans.brand_id!r}")
        except Exception as exc:  # never let one answer sink the run
            return _error_result(ans.answer_id, "internal_error", f"{type(exc).__name__}: {exc}"[:300])

    t0 = time.monotonic()
    status, run_error = "completed", None
    results: list[AgentResult] = []
    try:
        results = sorted(await asyncio.gather(*(one(a) for a in scheduled)), key=lambda r: r.answer_id)
    except Exception as exc:  # answers are isolated, so this means the run itself broke
        status, run_error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    report = build_report(meta, scheduled, results, gate, started_at, time.monotonic() - t0,
                          partial=status == "failed")
    db_answers = report.pop("db_answers")
    report.update(status=status, run_error=run_error, persistence_errors=[])
    out_dir = None
    if write_report:
        # Predictions reach disk before any network side effect, so a dead DB or tracing link cannot lose them.
        out_dir = write_run_files(report, runs_dir or config.RUNS_DIR)
        report["report_path"] = str(out_dir)

    if split == "development" and status == "completed":
        try:
            _attach_langfuse_scores(tracer, report, meta)
        except Exception as exc:
            report["persistence_errors"].append(f"langfuse scores: {type(exc).__name__}: {exc}"[:300])
    if store is not None:
        try:
            if status == "completed":
                await store.save_answers(run_id, db_answers)
            await store.finish_run(run_id, status, report["counts"], report["metrics"], run_error)
        except Exception as exc:
            report["persistence_errors"].append(f"database: {type(exc).__name__}: {exc}"[:300])
            report["status"] = "failed"
            report["run_error"] = run_error or "database write failed; report.json holds every prediction"
            try:
                await store.finish_run(run_id, "failed", report["counts"], report["metrics"], report["run_error"])
            except Exception:
                pass
    try:
        tracer.flush()
    except Exception as exc:
        report["persistence_errors"].append(f"langfuse flush: {type(exc).__name__}: {exc}"[:300])
    if out_dir is not None:
        write_run_files(report, out_dir.parent, out_dir=out_dir)
    return report


def db_rows_from_report(report: dict) -> list[dict]:
    keys = ("answer_id", "status", "error_type", "error_message", "latency_ms", "model_turns",
            "provider_attempts", "usage", "trace_id", "scores", "claims")
    return [{k: a.get(k) for k in keys} for a in report["answers"]]


async def import_report(store, report: dict) -> str:
    """Load a saved report into PostgreSQL without calling the model again."""
    run_id = uuid.UUID(report["run_id"])
    if report["status"] == "failed" and not any(e.startswith("database") for e in report.get("persistence_errors", [])):
        raise ValueError("report is from a run that failed before persistence; nothing complete to import")
    existing = await store.get_run(run_id)
    if existing is None:
        await store.insert_run({
            "run_id": run_id, "split": report["split"], "agent_version": report["agent_version"],
            **report["versions"], "provider": report["provider"], "model": report["model"],
            "settings": report["settings"], "file_hashes": report["file_hashes"],
            "source_revision": report["source_revision"]})
    elif existing["answers"]:
        raise ValueError(f"run {run_id} already has answer rows; refusing to import twice")
    await store.save_answers(run_id, db_rows_from_report(report))
    note = "answers imported from report.json after the original database write failed"
    await store.finish_run(run_id, "completed", report["counts"], report["metrics"], note)
    return note


def build_report(meta: dict, scheduled, results: list[AgentResult], gate: ProviderGate, started_at, duration_s: float,
                 partial: bool = False) -> dict:
    split = meta["split"]
    texts = {a.answer_id: a.ai_answer for a in scheduled}
    outcomes = [AnswerOutcome(r.answer_id, r.status, tuple(
        PredictedClaim(r.answer_id, c.claim_id, c.start_char, c.end_char, c.quote, c.label)
        for c in (r.output.claims if r.output else []))) for r in results]
    metrics, per_answer = None, {}
    gold_by_span = {}
    if split == "development" and scheduled and not partial:
        gold = list(load_gold())
        metrics = score_split([a.answer_id for a in scheduled], outcomes, gold, texts)
        per_answer = metrics.pop("per_answer")
        gold_by_span = {(g.answer_id, g.start_char, g.end_char): g for g in gold}

    answers_out, db_answers = [], []
    for r in results:
        claims = []
        for c in (r.output.claims if r.output else []):
            row = c.model_dump()
            if split == "development" and metrics is not None:
                g = gold_by_span.get((r.answer_id, c.start_char, c.end_char))
                row.update(matched_gold=g is not None, expected_label=g.expected_label if g else None,
                           label_correct=(g.expected_label == c.label) if g else None)
            claims.append(row)
        pa = per_answer.get(r.answer_id)
        scores = None if pa is None else {k: pa[k] for k in ("extraction_f1", "joint_f1", "gold", "predicted",
                                                              "matched", "correct", "missing_gold_ids",
                                                              "extra_claim_ids", "wrong_label")}
        base = {
            "answer_id": r.answer_id, "status": r.status,
            "error_type": r.error.error_type if r.error else None,
            "error_message": r.error.message if r.error else None,
            "latency_ms": r.latency_ms, "model_turns": r.turns, "provider_attempts": r.provider_attempts,
            "usage": r.usage, "trace_id": r.trace_id, "scores": scores, "claims": claims,
        }
        db_answers.append(base)
        answers_out.append({**base, "label_results": pa.get("label_results") if pa else None,
                            "exchanges": r.exchanges})

    ok = [r for r in results if r.status == "ok"]
    counts = {
        "scheduled_answers": len(scheduled), "terminal_answers": len(results), "ok_answers": len(ok),
        "error_answers": len(results) - len(ok),
        "empty_outputs": sum(1 for r in ok if not r.output.claims),
        "accepted_predictions": sum(len(r.output.claims) for r in ok),
        "error_types": _count(r.error.error_type for r in results if r.error),
        "provider_attempts": gate.total_attempts, "peak_active_requests": gate.peak_active,
        "G": None, "M": None, "C": None,
    }
    if metrics is not None:
        counts.update({k: metrics["counts"][k] for k in ("G", "M", "C")})
    dev_records = None
    if metrics is not None:
        gold_ix = {g.claim_id: g for g in load_gold()}
        pred_ix = {c["claim_id"]: c for a in answers_out for c in a["claims"]}
        dev_records = {
            "missed_gold": [{"claim_id": i, "expected_label": gold_ix[i].expected_label, "quote": gold_ix[i].quote}
                            for i in metrics["missed_gold_ids"]],
            "extra_predictions": [{"claim_id": i, "label": pred_ix[i]["label"], "quote": pred_ix[i]["quote"]}
                                  for i in metrics["unmatched_prediction_ids"]],
            "wrong_labels": [{"claim_id": i, "expected_label": gold_ix[i].expected_label,
                              "predicted_label": pred_ix[i]["label"], "quote": gold_ix[i].quote,
                              "reason": pred_ix[i]["reason"], "fact_ids": pred_ix[i]["fact_ids"]}
                             for i in metrics["wrong_label_ids"]],
        }
    return {
        "run_id": str(meta["run_id"]), "split": split, "agent_version": meta["agent_version"],
        "versions": {k: meta[k] for k in ("prompt_version", "dataset_version", "evaluator_version", "fact_version")},
        "provider": meta["provider"], "model": meta["model"], "settings": meta["settings"],
        "file_hashes": meta["file_hashes"], "source_revision": meta["source_revision"],
        "started_at": started_at.isoformat(), "duration_s": round(duration_s, 1),
        "counts": counts, "metrics": metrics, "development_records": dev_records,
        "answers": answers_out, "db_answers": db_answers,
    }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _attach_langfuse_scores(tracer, report: dict, meta: dict) -> None:
    if not getattr(tracer, "enabled", False) or report["metrics"] is None:
        return
    for a in report["answers"]:
        s = a["scores"]
        if not a["trace_id"] or s is None:
            continue
        detail = {"missing_gold_ids": s["missing_gold_ids"], "extra_claim_ids": s["extra_claim_ids"],
                  "wrong_label": s["wrong_label"], "status": a["status"], "error_type": a["error_type"]}
        tracer.score(trace_id=a["trace_id"], name="extraction_f1", value=s["extraction_f1"], metadata=detail)
        tracer.score(trace_id=a["trace_id"], name="joint_f1", value=s["joint_f1"], metadata=detail)
        for lr in a["label_results"] or []:
            tracer.score(trace_id=a["trace_id"], name="label_correct", value=1 if lr["correct"] else 0,
                         data_type="BOOLEAN", comment=lr["claim_id"],
                         metadata={"claim_id": lr["claim_id"], "expected": lr["expected"], "predicted": lr["predicted"]})
    m = report["metrics"]
    with tracer.run_trace(name=f"evaluation-run {meta['agent_version']} {meta['split']}", session_id=str(meta["run_id"]),
                          tags=[meta["agent_version"], meta["split"], "run-summary"],
                          metadata={"run_id": str(meta["run_id"]), "agent_version": meta["agent_version"],
                                    "prompt_version": meta["prompt_version"], "split": meta["split"],
                                    "model": meta["model"], "provider": meta["provider"]},
                          input={"settings": meta["settings"], "file_hashes": meta["file_hashes"]}) as run_trace:
        run_trace.finish(output={"counts": report["counts"], "extraction": m["extraction"], "joint": m["joint"],
                                 "macro_f1": m["macro_f1"], "gold_claim_coverage": m["gold_claim_coverage"]})
        trace_id = run_trace.trace_id
    for name, value in (("extraction_f1", m["extraction"]["f1"]), ("joint_f1", m["joint"]["f1"]),
                        ("macro_f1", m["macro_f1"]), ("gold_claim_coverage", m["gold_claim_coverage"])):
        tracer.score(trace_id=trace_id, name=f"run_{name}", value=value,
                     metadata={"missed_gold_ids": m["missed_gold_ids"],
                               "unmatched_prediction_ids": m["unmatched_prediction_ids"],
                               "wrong_label_ids": m["wrong_label_ids"]} if name == "joint_f1" else None)


def write_run_files(report: dict, runs_dir: Path, out_dir: Path | None = None) -> Path:
    stamp = report["started_at"][:19].replace(":", "").replace("-", "")
    tag = "_subset" if report["settings"].get("answer_subset") else ""
    out = out_dir or runs_dir / f"{stamp}_{report['agent_version']}_{report['split']}{tag}_{report['run_id'][:8]}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out
