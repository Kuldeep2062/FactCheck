import csv
import io

import pytest

from factcheck.data import (ANSWERS_FILE, data_file_hashes, load_answers, load_facts, load_gold, parse_value,
                            read_csv)

ISSUED_HASHES = {
    "answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724",
    "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2",
    "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850",
}


def test_issued_files_unchanged():
    assert data_file_hashes() == ISSUED_HASHES


def test_quoted_commas_and_embedded_newlines(tmp_path):
    text = 'Price is INR 1,299, tax included.\n\nSecond paragraph, "quoted".'
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["answer_id", "split", "brand_id", "question", "ai_answer"])
    w.writerow(["X1", "development", "cleartrail", "q, with comma", text])
    p = tmp_path / "a.csv"
    p.write_bytes(buf.getvalue().encode("utf-8"))
    rows = read_csv(p)
    assert len(rows) == 1
    assert rows[0]["ai_answer"] == text
    assert rows[0]["question"] == "q, with comma"


def test_counts_and_splits():
    answers = load_answers()
    assert len(answers) == 50
    assert sum(a.split == "development" for a in answers) == 40
    assert sum(a.split == "check" for a in answers) == 10
    assert len(load_facts()) == 132
    assert len(load_gold()) == 376


def test_ai_answer_preserved_exactly_for_gold_offsets():
    texts = {a.answer_id: a.ai_answer for a in load_answers()}
    for g in load_gold():
        assert texts[g.answer_id][g.start_char:g.end_char] == g.quote
    assert "\n\n" in texts["A01"]  # paragraph breaks survive parsing


def test_raw_bytes_roundtrip():
    raw = ANSWERS_FILE.read_bytes().decode("utf-8")
    for a in load_answers():
        assert a.ai_answer.replace('"', '""') in raw


def test_typed_values():
    assert parse_value("false", "boolean") is False
    assert parse_value("true", "boolean") is True
    assert parse_value("1299", "number") == 1299
    assert parse_value("5.3", "string") == "5.3"
    with pytest.raises(ValueError):
        parse_value("False", "boolean")
    facts = {f["fact_id"]: f for f in load_facts()}
    assert facts["cleartrail.mini.wifi"]["value"] is False
    assert facts["cleartrail.mini.model_name"]["unit"] is None
