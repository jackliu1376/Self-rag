import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.exceptions import ForbiddenError
from app.repositories.db import init_schema
from app.repositories.session_repository import SessionRepository


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_session_is_readable_by_its_owner(session_factory) -> None:
    async with session_factory() as db:
        repo = SessionRepository(db)
        await repo.get_or_create("sess_owner", "u_a")
        await repo.append_message("sess_owner", "user", "我的订单")
        await db.commit()

    async with session_factory() as db:
        repo = SessionRepository(db)
        row = await repo.get_or_create("sess_owner", "u_a")
        assert row.user_id == "u_a"
        history = await repo.recent_messages("sess_owner")
        assert [message.content for message in history] == ["我的订单"]


@pytest.mark.asyncio
async def test_summary_is_readable_on_the_next_turn(session_factory) -> None:
    async with session_factory() as db:
        repo = SessionRepository(db)
        await repo.get_or_create("sess_summary", "u_a")
        await repo.save_summary("sess_summary", "用户咨询定制杯退货")
        await db.commit()

    async with session_factory() as db:
        row = await SessionRepository(db).get_or_create("sess_summary", "u_a")
        assert row.summary == "用户咨询定制杯退货"


@pytest.mark.asyncio
async def test_session_cannot_be_opened_by_another_user(session_factory) -> None:
    async with session_factory() as db:
        await SessionRepository(db).get_or_create("sess_owner", "u_a")
        await db.commit()

    async with session_factory() as db:
        with pytest.raises(ForbiddenError) as exc:
            await SessionRepository(db).get_or_create("sess_owner", "u_intruder")
        assert exc.value.http_status == 403
