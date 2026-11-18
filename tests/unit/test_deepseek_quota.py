import pytest

from app.config.settings import Settings
from app.providers.deepseek import DeepSeekProvider


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


class _FakeClient:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def post(self, url: str, **kwargs) -> _FakeResponse:
        return _FakeResponse(self._body)


def _settings() -> Settings:
    return Settings(
        app_env="development",
        deepseek_api_key="test-key",
        deepseek_base_url="http://test",
    )


def _body(*, quota: dict | None = None) -> dict:
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    if quota is not None:
        usage["quota"] = quota
    return {"choices": [{"message": {"content": "ok"}}], "usage": usage}


@pytest.mark.asyncio
async def test_generate_records_quota_from_usage() -> None:
    provider = DeepSeekProvider(
        _settings(),
        client=_FakeClient(_body(quota={"model": "m", "percent": 23, "reset_at": "1795071817"})),
    )
    response = await provider.generate([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert provider.quota == {"used_percent": 23, "reset_at": 1795071817}


@pytest.mark.asyncio
async def test_generate_without_quota_keeps_last_known_value() -> None:
    provider = DeepSeekProvider(_settings(), client=_FakeClient(_body()))
    await provider.generate([{"role": "user", "content": "hi"}])
    assert provider.quota is None
