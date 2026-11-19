import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.config.settings import clear_settings_cache
from app.main import create_app


def _client(tmp_path: Path) -> tuple[TestClient, Path]:
    target = tmp_path / "knowledge"
    shutil.copytree("knowledge", target)
    os.environ["KNOWLEDGE_DIR"] = str(target)
    clear_settings_cache()
    return TestClient(create_app()), target


def test_reload_upsert_and_delete_document(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    try:
        with client:
            reloaded = client.post("/api/v1/admin/knowledge/reload")
            assert reloaded.status_code == 200
            assert reloaded.json()["chunks"] > 0

            upserted = client.put(
                "/api/v1/admin/knowledge/documents/test_hot_reload.md",
                json={"content": "# 蓝鲸退货窗口\n蓝鲸商品的退货窗口为九天，超过需人工审核。"},
            )
            assert upserted.status_code == 200
            assert upserted.json()["documents"] >= 1

            answer = client.post(
                "/api/v1/chat",
                json={"message": "蓝鲸退货窗口的规则是什么", "user_id": "u_demo"},
            ).json()
            assert "蓝鲸" in answer["answer"]

            deleted = client.delete("/api/v1/admin/knowledge/documents/test_hot_reload.md")
            assert deleted.status_code == 200

            after = client.post(
                "/api/v1/chat",
                json={"message": "蓝鲸退货窗口的规则是什么", "user_id": "u_demo"},
            ).json()
            assert "蓝鲸" not in after["answer"]
    finally:
        os.environ.pop("KNOWLEDGE_DIR", None)
        clear_settings_cache()
