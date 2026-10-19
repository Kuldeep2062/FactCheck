import argparse
import asyncio
import json

from .db import Database
from .api import default_provider
from .config import Settings
from .runner import import_report, run_split
from .tracing import make_tracer


async def _run(args) -> None:
    settings = Settings()
    store = None
    if not args.no_db:
        store = await Database.connect()
        await store.ensure_schema()
    try:
        report = await run_split(args.agent_version, args.split, provider=default_provider(settings),
                                 settings=settings, tracer=make_tracer(), store=store,
                                 answer_ids=args.answers.split(",") if args.answers else None)
    finally:
        if store is not None:
            await store.close()
    summary = {k: report.get(k) for k in ("run_id", "status", "run_error", "persistence_errors", "counts",
                                          "report_path")}
    if report["metrics"]:
        m = report["metrics"]
        summary["metrics"] = {"extraction_f1": m["extraction"]["f1"], "joint_f1": m["joint"]["f1"],
                              "macro_f1": m["macro_f1"], "gold_claim_coverage": m["gold_claim_coverage"]}
    print(json.dumps(summary, indent=2))


async def _import(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    store = await Database.connect()
    try:
        await store.ensure_schema()
        print(await import_report(store, report), report["run_id"])
    finally:
        await store.close()


async def _init_db() -> None:
    store = await Database.connect()
    await store.ensure_schema()
    await store.close()
    print("schema ready")


def main() -> None:
    p = argparse.ArgumentParser(prog="factcheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one agent version over one split")
    r.add_argument("--agent-version", choices=["v1", "v2"], required=True)
    r.add_argument("--split", choices=["development", "check"], required=True)
    r.add_argument("--no-db", action="store_true", help="skip PostgreSQL persistence")
    r.add_argument("--answers", help="comma-separated answer ids for a labelled subset run, e.g. A01,A02")
    sub.add_parser("init-db", help="create tables")
    imp = sub.add_parser("import-report", help="store a saved report.json whose database write failed")
    imp.add_argument("report")
    c = sub.add_parser("compare", help="compare two development reports")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--out")
    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    elif args.cmd == "import-report":
        asyncio.run(_import(args.report))
    elif args.cmd == "init-db":
        asyncio.run(_init_db())
    elif args.cmd == "compare":
        from .compare import compare_reports
        text = compare_reports(args.baseline, args.candidate)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        print(text)


if __name__ == "__main__":
    main()
