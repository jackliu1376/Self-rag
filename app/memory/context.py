"""Compact conversation context for prompt injection."""

from app.schemas.chat import ChatMessage

MAX_RECENT_WITH_SUMMARY = 2
MAX_RECENT_WITHOUT_SUMMARY = 8
FOLLOW_UP_MAX_CHARS = 20


def format_memory_context(messages: list[ChatMessage], summary: str, *, max_chars: int) -> str:
    """Shortcut: character cap, not the tokenizer; switch when prompts grow."""
    parts: list[str] = []
    summary = summary.strip()
    if summary:
        parts.append(f"[历史摘要]\n{summary}")
    limit = MAX_RECENT_WITH_SUMMARY if summary else MAX_RECENT_WITHOUT_SUMMARY
    recent = [message for message in messages if message.content.strip()][-limit:]
    if recent:
        lines = [
            f"{'用户' if message.role == 'user' else '客服'}：{message.content.strip()}"
            for message in recent
        ]
        parts.append("[最近对话]\n" + "\n".join(lines))
    context = "\n\n".join(parts)
    return context[-max_chars:] if len(context) > max_chars else context


def retrieval_query(
    query: str, messages: list[ChatMessage], *, max_context_chars: int = 200
) -> str:
    """Short follow-ups need the previous user turn to retrieve the right policy."""
    if len(query.strip()) > FOLLOW_UP_MAX_CHARS:
        return query
    previous = next(
        (message.content.strip() for message in reversed(messages) if message.role == "user"),
        "",
    )
    if not previous:
        return query
    return f"{previous[:max_context_chars]} {query}"
