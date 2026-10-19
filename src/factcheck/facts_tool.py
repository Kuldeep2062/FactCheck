"""Read-only catalogue lookup exposed to the model as get_brand_facts."""
from functools import lru_cache

from .config import FACT_VERSION
from .data import load_facts

TOOL_NAME = "get_brand_facts"


class ToolError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


@lru_cache
def known_brands() -> tuple[str, ...]:
    return tuple(sorted({f["brand_id"] for f in load_facts()}))


@lru_cache
def all_fact_ids() -> frozenset[str]:
    return frozenset(f["fact_id"] for f in load_facts())


def get_brand_facts(brand_id: str) -> dict:
    if brand_id not in known_brands():
        raise ToolError("unknown_brand", f"unknown brand_id {brand_id!r}")
    return {
        "brand_id": brand_id,
        "fact_version": FACT_VERSION,
        "facts": [f for f in load_facts() if f["brand_id"] == brand_id],
    }


def tool_declaration_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "brand_id": {
                "type": "string",
                "enum": list(known_brands()),
                "description": "Brand whose catalogue rows to return.",
            }
        },
        "required": ["brand_id"],
    }


def execute_tool_call(name: str, args: dict) -> dict:
    """Validate a model-requested call before running it; nothing else is ever executed."""
    if name != TOOL_NAME:
        raise ToolError("illegal_tool", f"model requested unknown tool {name!r}")
    if not isinstance(args, dict) or set(args) != {"brand_id"} or not isinstance(args["brand_id"], str):
        raise ToolError("invalid_tool_arguments", f"invalid arguments for {TOOL_NAME}: {args!r}")
    return get_brand_facts(args["brand_id"])
