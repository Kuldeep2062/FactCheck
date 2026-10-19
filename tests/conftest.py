import asyncio
import copy
import json
import re

import pytest

from factcheck.config import Settings
from factcheck.provider import ModelRequest, ModelResponse, ToolCall
from factcheck.reliability import ProviderGate


def tool_response(*brands: str) -> ModelResponse:
    return ModelResponse(text=None, tool_calls=[ToolCall("get_brand_facts", {"brand_id": b}) for b in brands],
                         raw_content=object())


def final_response(claims: list[dict]) -> ModelResponse:
    return ModelResponse(text=json.dumps({"claims": claims}), tool_calls=[], raw_content=object())


class FakeProvider:
    """Scripted provider. `script` is either a list consumed in order or a callable(request, n) -> response."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script, delay: float = 0.0):
        self.script = script
        self.delay = delay
        self.requests: list[ModelRequest] = []
        self.active = 0
        self.peak = 0

    def parameters(self):
        return {}

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(ModelRequest(request.system, copy.deepcopy(
            [{k: v for k, v in m.items() if k != "raw"} for m in request.messages]), request.mode))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            n = len(self.requests)
            item = self.script(request, n) if callable(self.script) else self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                item = await item()
            return item
        finally:
            self.active -= 1


def sentence_ids_in(request: ModelRequest) -> list[int]:
    return [int(x) for x in re.findall(r"^\[S(\d+)\]", request.messages[0]["text"], re.M)]


async def no_sleep(_s: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _reports_to_tmp(tmp_path, monkeypatch):
    # Test runs must never write report folders into the real runs/ evidence directory.
    import factcheck.config
    monkeypatch.setattr(factcheck.config, "RUNS_DIR", tmp_path / "runs")


@pytest.fixture
def settings() -> Settings:
    return Settings(requests_per_minute=0, tokens_per_minute=0)


@pytest.fixture
def gate() -> ProviderGate:
    return ProviderGate(max_concurrent=3, timeout_s=1.0, max_attempts=2, retry_delay_s=0.0, sleep=no_sleep)
