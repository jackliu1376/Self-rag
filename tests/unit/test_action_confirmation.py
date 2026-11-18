import pytest

from app.agents.nodes.planner import Planner, planner_node
from app.core.enums import ToolRisk
from app.providers.fake import OfflineLLMProvider
from app.schemas.chat import ActionProposal
from app.tools.aftersale_tools import CancelOrderTool
from app.tools.order_tools import GetOrderTool
from app.tools.registry import ToolRegistry


def _action() -> ActionProposal:
    return ActionProposal(
        action_id="act_test",
        tool_name="cancel_order",
        arguments={"order_id": "ORD10003"},
        risk=ToolRisk.DESTRUCTIVE,
        summary="确认后将取消订单 ORD10003。",
    )


@pytest.mark.asyncio
async def test_cancellation_needs_explicit_request_and_confirmation() -> None:
    registry = ToolRegistry()
    registry.register(GetOrderTool())
    registry.register(CancelOrderTool())
    planner = Planner(OfflineLLMProvider(), registry)

    proposal = await planner_node({"user_query": "取消订单 ORD10003"}, planner)
    assert proposal["pending_action"].tool_name == "cancel_order"
    assert "cancel_order" not in proposal["pending_action"].summary

    question = await planner_node({"user_query": "取消订单有什么规则？"}, planner)
    assert question["pending_action"] is None

    confirmed = await planner_node(
        {"user_query": "确认执行", "confirmed_action": _action()}, planner
    )
    assert len(confirmed["plan"].steps) == 1
    assert confirmed["plan"].steps[0].arguments == {"order_id": "ORD10003"}
