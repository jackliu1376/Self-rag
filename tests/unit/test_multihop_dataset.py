import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "dataset" / "rag_eval_multihop.jsonl"


def test_multihop_rows_require_at_least_two_documents() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) >= 15
    for row in rows:
        assert len(row["relevant_doc_ids"]) >= 2, row["query"]
        assert row["gold_answer_contains"], row["query"]


def test_multihop_documents_exist_in_knowledge() -> None:
    knowledge = ROOT / "knowledge"
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        for document_id in row["relevant_doc_ids"]:
            assert (knowledge / f"{document_id}.md").is_file(), document_id
