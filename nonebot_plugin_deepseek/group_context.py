from importlib.util import find_spec
from typing import Optional

from nonebot import require
from nonebot.adapters import Event
from nonebot.matcher import current_bot

from .config import ds_config

if find_spec("nonebot_plugin_uninfo"):
    require("nonebot_plugin_uninfo")
    uninfo_enable = True
else:
    uninfo_enable = False


# {session_id: [message, ...]}
_group_contexts: dict[str, list[dict[str, str]]] = {}


async def get_group_session_id(event: Event) -> Optional[str]:
    """Return a session id for group/guild chats, or None for private chats."""
    if uninfo_enable:
        try:
            from nonebot_plugin_uninfo import get_session, SceneType

            session = await get_session(current_bot.get(), event)
            if session is None:
                return None
            scene = session.scene
            if scene.type == SceneType.GROUP:
                return f"{session.scope}:group:{scene.id}"
            if scene.type == SceneType.GUILD:
                channel = scene.parent.id if scene.parent else scene.id
                return f"{session.scope}:guild:{channel}"
        except Exception:
            pass

    # Fallback for common adapters
    message_type = getattr(event, "message_type", None)
    if message_type == "group":
        group_id = getattr(event, "group_id", None)
        if group_id is not None:
            return f"fallback:group:{group_id}"
    return None


async def get_user_name(event: Event) -> str:
    user_id = event.get_user_id()
    if uninfo_enable:
        try:
            from nonebot_plugin_uninfo import get_session

            session = await get_session(current_bot.get(), event)
            if session is None:
                return user_id
            if session.user.name:
                return session.user.name
            return session.user.id
        except Exception:
            pass
    try:
        sender = getattr(event, "sender", None)
        if sender:
            nickname = getattr(sender, "nickname", None)
            if nickname:
                return str(nickname)
    except Exception:
        pass
    return user_id


def get_context(session_id: str) -> list[dict[str, str]]:
    return _group_contexts.get(session_id, [])


def set_context(session_id: str, context: list[dict[str, str]]) -> None:
    limit = ds_config.max_group_history
    if len(context) > limit:
        context = context[-limit:]
    _group_contexts[session_id] = context


def clear_context(session_id: str) -> None:
    _group_contexts.pop(session_id, None)
