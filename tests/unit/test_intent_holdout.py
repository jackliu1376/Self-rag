import json
from pathlib import Path

from app.core.enums import IntentLabel

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "eval" / "dataset" / "intent_holdout.jsonl"
TRAIN = ROOT / "eval" / "dataset" / "intent_100.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_holdout_is_labeled_and_disjoint_from_training_set() -> None:
    rows = _rows(HOLDOUT)
    train_queries = {row["query"] for row in _rows(TRAIN)}
    assert len(rows) >= 40
    assert len({row["query"] for row in rows}) == len(rows)
    assert not ({row["query"] for row in rows} & train_queries)
    labels = {label.value for label in IntentLabel}
    for row in rows:
        assert row["intent"] in labels, row["query"]
