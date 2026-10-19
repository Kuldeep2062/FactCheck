from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["supported", "contradicted", "unknown"]
AgentVersion = Literal["v1", "v2"]


class AgentInput(BaseModel):
    # extra="forbid" makes any smuggled gold/claim-list field a 422 at the API boundary.
    model_config = ConfigDict(extra="forbid")
    answer_id: str = Field(min_length=1, max_length=64)
    brand_id: str = Field(min_length=1, max_length=64)
    question: str
    ai_answer: str = Field(min_length=1)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str
    label: Label
    fact_ids: list[str]
    reason: str


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_id: str
    claims: list[Claim]


class ModelClaim(BaseModel):
    """What the model returns per factual sentence; offsets are derived in Python."""
    model_config = ConfigDict(extra="forbid")
    sentence_id: int
    label: Label
    fact_ids: list[str]
    reason: str


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ModelClaim]


MODEL_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_id": {"type": "integer", "description": "Number of the factual sentence, e.g. 4 for [S4]."},
                    "label": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
                    "fact_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "description": "One short sentence."},
                },
                "required": ["sentence_id", "label", "fact_ids", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class AgentError(BaseModel):
    error_type: str
    message: str
