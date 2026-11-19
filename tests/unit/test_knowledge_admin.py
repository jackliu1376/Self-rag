from pathlib import Path

import pytest

from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError
from app.retrieval.cache import MemoryRetrievalCache
from app.retrieval.hybrid import HybridRetriever


def _settings(tmp_path: Path) -> Settings:
    return Settings(app_env="testing", knowledge_dir=str(tmp_path))


def _resolve(tmp_path: Path, doc_path: str) -> Path:
    from app.api.routes.admin import resolve_document_path

    return resolve_document_path(_settings(tmp_path), doc_path)


def test_accepts_flat_markdown_document(tmp_path: Path) -> None:
    assert _resolve(tmp_path, "new_policy.md") == (tmp_path / "new_policy.md").resolve()


@pytest.mark.parametrize(
    "bad_path",
    ["../evil.md", "sub/dir.md", "notes.txt", ".hidden.md", "", "/abs.md"],
)
def test_rejects_unsafe_or_unsupported_paths(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(InvalidRequestError):
        _resolve(tmp_path, bad_path)


@pytest.mark.asyncio
async def test_memory_cache_clear_drops_entries() -> None:
    cache = MemoryRetrievalCache(ttl_seconds=60)
    await cache.set("退货政策", 3, [])
    assert await cache.get("退货政策", 3) == []
    await cache.clear()
    assert await cache.get("退货政策", 3) is None


def test_replace_index_swaps_bm25_and_vector() -> None:
    class _Component:
        pass

    old_bm25, old_vector = _Component(), _Component()
    new_bm25, new_vector = _Component(), _Component()
    retriever = HybridRetriever(old_bm25, old_vector, rrf_k=60)  # type: ignore[arg-type]
    retriever.replace_index(new_bm25, new_vector)  # type: ignore[arg-type]
    assert retriever._bm25 is new_bm25
    assert retriever._vector is new_vector
