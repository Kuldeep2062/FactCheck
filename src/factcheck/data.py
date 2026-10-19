import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR

ANSWERS_FILE = DATA_DIR / "answers.csv"
FACTS_FILE = DATA_DIR / "brand_facts.csv"
GOLD_FILE = DATA_DIR / "development_gold.csv"


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: tuple[str, ...]


def read_csv(path: Path) -> list[dict[str, str]]:
    # newline="" keeps embedded line breaks inside quoted fields byte-exact.
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_value(raw: str, value_type: str):
    if value_type == "number":
        num = float(raw)
        return int(num) if num.is_integer() and "." not in raw else num
    if value_type == "boolean":
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise ValueError(f"invalid boolean literal {raw!r}")
    if value_type == "string":
        return raw
    raise ValueError(f"unknown value_type {value_type!r}")


def parse_fact_row(row: dict[str, str]) -> dict:
    return {
        "fact_id": row["fact_id"],
        "brand_id": row["brand_id"],
        "model_id": row["model_id"],
        "field": row["field"],
        "value": parse_value(row["value"], row["value_type"]),
        "value_type": row["value_type"],
        "unit": row["unit"] or None,
        "condition": row["condition"] or None,
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
    }


@lru_cache
def load_answers(path: Path = ANSWERS_FILE) -> tuple[Answer, ...]:
    return tuple(
        Answer(r["answer_id"], r["split"], r["brand_id"], r["question"], r["ai_answer"])
        for r in read_csv(path)
    )


@lru_cache
def load_facts(path: Path = FACTS_FILE) -> tuple[dict, ...]:
    return tuple(parse_fact_row(r) for r in read_csv(path))


@lru_cache
def load_gold(path: Path = GOLD_FILE) -> tuple[GoldClaim, ...]:
    return tuple(
        GoldClaim(
            r["answer_id"], r["claim_id"], int(r["start_char"]), int(r["end_char"]),
            r["quote"], r["expected_label"],
            tuple(x for x in r["evidence_fact_ids"].split("|") if x),
        )
        for r in read_csv(path)
    )


def answers_for_split(split: str) -> list[Answer]:
    return sorted((a for a in load_answers() if a.split == split), key=lambda a: a.answer_id)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_file_hashes() -> dict[str, str]:
    return {p.name: file_sha256(p) for p in (ANSWERS_FILE, FACTS_FILE, GOLD_FILE)}
