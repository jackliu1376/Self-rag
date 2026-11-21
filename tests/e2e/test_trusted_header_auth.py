import os

from fastapi.testclient import TestClient

from app.config.settings import clear_settings_cache
from app.main import create_app


def _client(mode: str) -> TestClient:
    os.environ["AUTH_MODE"] = mode
    clear_settings_cache()
    return TestClient(create_app())


def test_trusted_header_auth() -> None:
    try:
        with _client("trusted_header") as client:
            missing = client.post("/api/v1/chat", json={"message": "你好", "user_id": "u_demo"})
            assert missing.status_code == 401
            assert missing.json()["error_code"] == "UNAUTHORIZED"

            too_long = client.post(
                "/api/v1/chat",
                json={"message": "你好"},
                headers={"X-User-Id": "x" * 65},
            )
            assert too_long.status_code == 400

            created = client.post(
                "/api/v1/chat",
                json={"message": "你好", "user_id": "u_demo"},
                headers={"X-User-Id": "alice"},
            )
            assert created.status_code == 200
            session_id = created.json()["session_id"]

            forbidden = client.post(
                "/api/v1/chat",
                json={"message": "你好", "user_id": "u_demo", "session_id": session_id},
                headers={"X-User-Id": "bob"},
            )
            assert forbidden.status_code == 403
            assert forbidden.json()["error_code"] == "FORBIDDEN"

        with _client("body") as client:
            default = client.post("/api/v1/chat", json={"message": "你好", "user_id": "u_demo"})
            assert default.status_code == 200
    finally:
        os.environ.pop("AUTH_MODE", None)
        clear_settings_cache()
