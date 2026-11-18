from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.enums import ToolRisk
from app.core.exceptions import InvalidRequestError
from app.repositories.db import init_schema
from app.repositories.pending_actions import PendingActionRepository
from app.schemas.chat import ActionProposal, ChatRequest, ChatResponse
from app.services.chat_service import ChatService


def _action(action_id: str = "act_test") -> ActionProposal:
    return ActionProposal(
        action_id=action_id,
        tool_name="cancel_order",
        arguments={"order_id": "ORD10003"},
        risk=ToolRisk.DESTRUCTIVE,
        summary="确认后将取消订单 ORD10003。",
    )


async def _factory(url: str) -> tuple:
    engine = create_async_engine(url)
    await init_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, factory


@pytest.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine, factory = await _factory(f"sqlite+aiosqlite:///{tmp_path / 'pending.db'}")
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_pending_action_is_readable_from_another_connection(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}"
    engine_a, factory_a = await _factory(url)
    async with factory_a() as db:
        await PendingActionRepository(db).put(
            _action(), session_id="sess_a", user_id="u_a", ttl_seconds=900
        )
        await db.commit()
    await engine_a.dispose()

    engine_b, factory_b = await _factory(url)
    async with factory_b() as db:
        action = await PendingActionRepository(db).get(
            "act_test", session_id="sess_a", user_id="u_a"
        )
        assert action == _action()
    await engine_b.dispose()


@pytest.mark.asyncio
async def test_pending_action_is_bound_to_session_and_user(factory) -> None:
    async with factory() as db:
        repo = PendingActionRepository(db)
        await repo.put(_action(), session_id="sess_a", user_id="u_a", ttl_seconds=900)
        await db.commit()

    async with factory() as db:
        repo = PendingActionRepository(db)
        assert await repo.get("act_test", session_id="sess_a", user_id="u_a") == _action()
        assert await repo.get("act_test", session_id="sess_b", user_id="u_a") is None
        assert await repo.get("act_test", session_id="sess_a", user_id="u_b") is None


@pytest.mark.asyncio
async def test_expired_pending_action_is_rejected(factory) -> None:
    async with factory() as db:
        await PendingActionRepository(db).put(
            _action(), session_id="sess_a", user_id="u_a", ttl_seconds=-1
        )
        await db.commit()

    async with factory() as db:
        action = await PendingActionRepository(db).get(
            "act_test", session_id="sess_a", user_id="u_a"
        )
        assert action is None


@pytest.mark.asyncio
async def test_service_persists_and_resolves_confirmation(factory) -> None:
    service = ChatService.__new__(ChatService)
    request = ChatRequest(message="取消订单 ORD10003", user_id="u_demo")
    response = ChatResponse(
        answer="请确认",
        session_id="sess_demo",
        trace_id="trace_demo",
        pending_action=_action(),
    )
    async with factory() as db:
        await service._remember_pending(db, request, response)
        await db.commit()

    confirmed = ChatRequest(
        message="确认执行",
        user_id="u_demo",
        session_id="sess_demo",
        confirm_action_id="act_test",
    )
    async with factory() as db:
        assert await service._resolve_confirmation(db, confirmed, "sess_demo") == _action()
        with pytest.raises(InvalidRequestError):
            await service._resolve_confirmation(
                db, confirmed.model_copy(update={"user_id": "someone_else"}), "sess_demo"
            )
        with pytest.raises(InvalidRequestError):
            await service._resolve_confirmation(db, confirmed, "another_session")
