import inspect
from typing import Union

from nonebot.typing import T_State
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_alconna.extension import Extension
from nonebot.internal.adapter import Bot, Event, Message
from nonebot_plugin_alconna import Arparma, OptionResult


class CleanDocExtension(Extension):
    @property
    def priority(self) -> int:
        return 15

    @property
    def id(self) -> str:
        return "CleanDoc"

    async def send_wrapper(self, bot: Bot, event: Event, send: Union[str, Message, UniMessage]):
        plain_text = send if isinstance(send, (Message, UniMessage)) else inspect.cleandoc(send)
        return plain_text


class ParseExtension(Extension):
    @property
    def priority(self) -> int:
        return 20

    @property
    def id(self) -> str:
        return "ParseExtension"

    async def parse_wrapper(self, bot: Bot, state: T_State, event: Event, res: Arparma) -> None:
        if res.subcommands.get("model") and not res.subcommands["model"].options:
            res.subcommands["model"].options.setdefault("list", OptionResult())
        elif res.subcommands.get("tts") and not res.subcommands["tts"].options:
            res.subcommands["tts"].options.setdefault("list", OptionResult())

        return None
