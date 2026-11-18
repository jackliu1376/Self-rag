"""DB-backed pending confirmations for high-risk tool calls.

Shared through the database so confirmations survive restarts and work
across multiple API instances.
"""

from __future__ import annotations

import json
import time

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ToolRisk
from app.repositories.models import PendingActionRow
from app.schemas.chat import ActionProposal


class PendingActionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def put(
        self,
        action: ActionProposal,
        *,
        session_id: str,
        user_id: str,
        ttl_seconds: int,
    ) -> None:
        now = int(time.time())
        await self._session.execute(
            delete(PendingActionRow).where(PendingActionRow.expires_at <= now)
        )
        if await self._session.get(PendingActionRow, action.action_id) is not None:
            return
        self._session.add(
            PendingActionRow(
                action_id=action.action_id,
                session_id=session_id,
                user_id=user_id,
                tool_name=action.tool_name,
                arguments_json=json.dumps(action.arguments, ensure_ascii=False),
                risk=action.risk.value,
                summary=action.summary,
                expires_at=now + ttl_seconds,
            )
        )
        await self._session.flush()

    async def get(self, action_id: str, *, session_id: str, user_id: str) -> ActionProposal | None:
        row = await self._session.get(PendingActionRow, action_id)
        if row is None or row.session_id != session_id or row.user_id != user_id:
            return None
        if row.expires_at <= int(time.time()):
            await self._session.delete(row)
            return None
        return ActionProposal(
            action_id=row.action_id,
            tool_name=row.tool_name,
            arguments=json.loads(row.arguments_json),
            risk=ToolRisk(row.risk),
            summary=row.summary,
        )
