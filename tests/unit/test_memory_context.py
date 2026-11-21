from app.memory.context import format_memory_context, retrieval_query
from app.schemas.chat import ChatMessage


def _messages(count: int) -> list[ChatMessage]:
    return [
        ChatMessage(role="user" if index % 2 == 0 else "assistant", content=f"m{index}")
        for index in range(count)
    ]


def test_includes_summary_and_recent_messages() -> None:
    context = format_memory_context(
        [ChatMessage(role="user", content="你好"), ChatMessage(role="assistant", content="您好")],
        "用户咨询退货",
        max_chars=500,
    )
    assert "[历史摘要]" in context
    assert "用户咨询退货" in context
    assert "用户：你好" in context
    assert "客服：您好" in context


def test_summary_limits_raw_history_to_the_tail() -> None:
    context = format_memory_context(_messages(10), "摘要", max_chars=500)
    assert "m0" not in context
    assert "m8" in context
    assert "m9" in context


def test_without_summary_keeps_more_recent_messages() -> None:
    context = format_memory_context(_messages(10), "", max_chars=500)
    assert "m2" in context
    assert "m1" not in context


def test_caps_context_length() -> None:
    context = format_memory_context(_messages(10), "很长的摘要" * 20, max_chars=20)
    assert len(context) <= 20


def test_empty_history_returns_empty_context() -> None:
    assert format_memory_context([], "", max_chars=100) == ""


def test_short_follow_up_uses_previous_user_turn_for_retrieval() -> None:
    messages = [
        ChatMessage(role="user", content="我想退定制马克杯，杯身有裂纹"),
        ChatMessage(role="assistant", content="定制商品质量问题可以退。"),
    ]
    query = retrieval_query("那还能退吗？", messages)
    assert "定制马克杯" in query
    assert "那还能退吗" in query


def test_long_query_is_unchanged_for_retrieval() -> None:
    query = "定制马克杯有裂纹还能退吗请给出政策依据和具体步骤"
    assert retrieval_query(query, [ChatMessage(role="user", content="之前的问题")]) == query


def test_without_history_keeps_original_retrieval_query() -> None:
    assert retrieval_query("那还能退吗？", []) == "那还能退吗？"
