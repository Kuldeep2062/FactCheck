"""Thin FastAPI layer; all logic lives in agent, runner and db modules."""
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .db import Database
from .agent import Agent, UnknownPrimaryBrand
from .config import Settings
from .provider import make_provider
from .runner import make_gate, run_split
from .schemas import AgentInput, AgentOutput, AgentVersion
from .tracing import make_tracer


class ClassifyRequest(AgentInput):
    agent_version: AgentVersion


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: AgentVersion
    split: Literal["development", "check"]


def default_provider(settings: Settings):
    return make_provider(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = app.state
    if not hasattr(st, "settings"):
        st.settings = Settings()
    if not hasattr(st, "provider"):
        st.provider = default_provider(st.settings)
    if not hasattr(st, "tracer"):
        st.tracer = make_tracer()
    if not hasattr(st, "gate"):
        st.gate = make_gate(st.settings)
    owns_store = False
    if not hasattr(st, "store"):
        st.store = await Database.connect() if os.getenv("DATABASE_URL") else None
        owns_store = st.store is not None
    if st.store is not None:
        await st.store.ensure_schema()
    yield
    if owns_store:
        await st.store.close()
    st.tracer.flush()


app = FastAPI(title="factcheck", lifespan=lifespan)


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": error_type, "message": message}})


@app.post("/classify", response_model=AgentOutput)
async def classify(body: ClassifyRequest, request: Request):
    st = request.app.state
    agent = Agent(body.agent_version, st.provider, st.gate, st.settings)
    inp = AgentInput(**body.model_dump(exclude={"agent_version"}))
    try:
        result = await agent.run(inp, st.tracer, {"split": "adhoc"})
    except UnknownPrimaryBrand:
        return _error(404, "unknown_brand", f"unknown primary brand_id {body.brand_id!r}")
    if result.status == "ok":
        return result.output
    err = result.error
    status = 504 if err.error_type == "provider_timeout" else 502
    return _error(status, err.error_type, err.message)


@app.post("/evaluation-runs", status_code=201)
async def create_evaluation_run(body: EvaluationRunRequest, request: Request):
    st = request.app.state
    if st.store is None:
        raise HTTPException(503, "DATABASE_URL is not configured")
    report = await run_split(body.agent_version, body.split, provider=st.provider, settings=st.settings,
                             tracer=st.tracer, store=st.store)
    return {"run_id": report["run_id"], "status": report["status"], "counts": report["counts"],
            "metrics": report["metrics"], "report_path": report.get("report_path")}


@app.get("/evaluation-runs/{run_id}")
async def get_evaluation_run(run_id: uuid.UUID, request: Request):
    st = request.app.state
    if st.store is None:
        raise HTTPException(503, "DATABASE_URL is not configured")
    run = await st.store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run
