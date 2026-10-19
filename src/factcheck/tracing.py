"""Thin Langfuse wrapper. NullTracer keeps tests and key-less runs offline."""
import os
from contextlib import contextmanager
from typing import Any, Iterator


class _NullObs:
    trace_id = None
    id = None

    def update(self, **_: Any) -> "_NullObs":
        return self

    def end(self) -> None:
        pass


class NullAnswerTrace:
    trace_id: str | None = None

    def generation(self, **_: Any) -> _NullObs:
        return _NullObs()

    def tool(self, **_: Any) -> _NullObs:
        return _NullObs()

    def finish(self, **_: Any) -> None:
        pass


class NullTracer:
    enabled = False

    @contextmanager
    def answer_trace(self, **_: Any) -> Iterator[NullAnswerTrace]:
        yield NullAnswerTrace()

    @contextmanager
    def run_trace(self, **_: Any) -> Iterator[NullAnswerTrace]:
        yield NullAnswerTrace()

    def score(self, **_: Any) -> None:
        pass

    def trace_url(self, trace_id: str | None) -> str | None:
        return None

    def flush(self) -> None:
        pass


class LangfuseAnswerTrace:
    def __init__(self, root):
        self._root = root
        self.trace_id: str | None = root.trace_id

    def generation(self, *, name: str, model: str, input: Any, model_parameters: dict, metadata: dict):
        return self._root.start_observation(name=name, as_type="generation", model=model, input=input,
                                            model_parameters=model_parameters, metadata=metadata)

    def tool(self, *, name: str, input: Any):
        return self._root.start_observation(name=name, as_type="tool", input=input)

    def finish(self, *, output: Any, metadata: dict | None = None, level: str | None = None,
               status_message: str | None = None) -> None:
        self._root.update(output=output, metadata=metadata, level=level, status_message=status_message)


def _attr_metadata(metadata: dict) -> dict[str, str]:
    # Propagated trace attributes must be short strings.
    return {k: str(v)[:200] for k, v in metadata.items() if v is not None}


class LangfuseTracer:
    enabled = True

    def __init__(self):
        from langfuse import Langfuse
        self.client = Langfuse()

    @contextmanager
    def _trace(self, *, name: str, session_id: str, tags: list[str], metadata: dict, input: Any, as_type: str):
        from langfuse import propagate_attributes
        with propagate_attributes(session_id=session_id, tags=tags, trace_name=name,
                                  metadata=_attr_metadata(metadata)):
            with self.client.start_as_current_observation(name=name, as_type=as_type, input=input,
                                                          metadata=metadata) as root:
                yield LangfuseAnswerTrace(root)

    def answer_trace(self, **kw):
        return self._trace(as_type="agent", **kw)

    def run_trace(self, **kw):
        return self._trace(as_type="chain", **kw)

    def score(self, *, trace_id: str | None, name: str, value: float | str, data_type: str = "NUMERIC",
              comment: str | None = None, metadata: dict | None = None) -> None:
        if trace_id:
            self.client.create_score(trace_id=trace_id, name=name, value=value, data_type=data_type,
                                     comment=comment, metadata=metadata)

    def trace_url(self, trace_id: str | None) -> str | None:
        return self.client.get_trace_url(trace_id=trace_id) if trace_id else None

    def flush(self) -> None:
        self.client.flush()


def make_tracer(enabled: bool | None = None):
    if enabled is None:
        enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return LangfuseTracer() if enabled else NullTracer()
