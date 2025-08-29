import re
import importlib
from dataclasses import asdict
from collections.abc import Callable, Awaitable
from typing import Any, Union, Literal, Optional

import httpx
from nonebot.adapters import Event
from nonebot.permission import User, Permission
from nonebot_plugin_waiter import Waiter, prompt
from nonebot_plugin_alconna import SupportAdapter
from nonebot.matcher import Matcher, current_event, current_matcher
from nonebot_plugin_alconna.uniseg import UniMsg, UniMessage, get_target, get_message_id, message_reaction

from .apis import API
from .log import tts_logger
from .schemas import Message
from .exception import RequestException
from .function_call.registry import registry
from .config import CustomTTS, CustomModel, config


class DeepSeekHandler:
    def __init__(
        self,
        model: CustomModel,
        is_to_pic: bool,
        is_use_tts: bool,
        is_contextual: bool,
        tts_model: Optional[CustomTTS] = None,
    ) -> None:
        self.model: CustomModel = model
        self.is_to_pic: bool = is_to_pic
        self.is_use_tts: bool = is_use_tts
        self.is_contextual: bool = is_contextual
        self.tts_model: Optional[CustomTTS] = tts_model
        self.event: Event = current_event.get()
        self.matcher: Matcher = current_matcher.get()
        self.message_id: str = get_message_id(self.event)
        self.waiter: Waiter[Union[str, Literal[False]]] = self._setup_waiter()

        self.context: list[dict[str, Any]] = []

        self.md_to_pic: Union[Callable[..., Awaitable[bytes]], None] = (
            importlib.import_module("nonebot_plugin_htmlrender").md_to_pic if self.is_to_pic else None
        )

    async def handle(self, content: Optional[str]) -> None:
        if content:
            self.context.append({"role": "user", "content": content})

        await self._message_reaction("thinking")

        if not self.is_contextual:
            await self._handle_single_conversion()
        else:
            await self._handle_multi_round_conversion()

    async def _handle_single_conversion(self) -> None:
        if message := await self._get_response_message():
            await self._send_response(message)

    async def _handle_multi_round_conversion(self) -> None:
        timeout = config.timeout if isinstance(config.timeout, int) else config.timeout.user_input
        async for resp in self.waiter(default=False, timeout=timeout):
            await self._process_waiter_response(resp)

            if resp == "rollback":
                continue

            message = await self._get_response_message()
            if not message:
                continue

            await self._send_response(message)
            self.context.append(asdict(message))

            if await self._handle_tool_calls(message):
                self.waiter.future.set_result("")
                continue

    def _setup_waiter(self) -> Waiter[Union[str, Literal[False]]]:
        permission = Permission(User.from_event(self.event, perm=self.matcher.permission))
        waiter = Waiter(
            waits=["message"],
            handler=self._waiter_handler,
            matcher=self.matcher,
            permission=permission,
        )
        waiter.future.set_result("")
        return waiter

    def _waiter_handler(self, msg: UniMsg, skip: bool = False) -> Union[str, Literal[False]]:
        text = msg.extract_plain_text()
        if not skip:
            self.message_id = get_message_id()
        if text in ["结束", "取消", "done"]:
            return False
        if text in ["回滚", "rollback"]:
            return "rollback"
        return text

    def _prompt_handler(self, msg: UniMsg) -> UniMsg:
        self.message_id = get_message_id()
        return msg

    async def _message_reaction(self, status: Literal["fail", "thinking", "done"]) -> None:
        emoji_map = {
            "fail": ["10060", "❌"],
            "thinking": ["424", "👀"],
            "done": ["144", "🎉"],
        }
        target = get_target(self.event)
        if target.adapter == SupportAdapter.onebot11:
            emoji = emoji_map[status][0]
        else:
            emoji = emoji_map[status][1]

        await message_reaction(emoji, message_id=self.message_id)

    async def _process_waiter_response(self, resp: Union[bool, str]) -> None:
        timeout = config.timeout if isinstance(config.timeout, int) else config.timeout.user_input

        if resp == "" and not self.context:
            _resp = await prompt(
                "你想对 DeepSeek 说什么呢？",
                handler=self._prompt_handler,
                timeout=timeout,
            )
            if _resp is None:
                await UniMessage.text("等待超时").finish(reply_to=self.message_id)
            resp = self._waiter_handler(_resp, skip=True)

        await self._message_reaction("thinking")

        if resp is False:
            await UniMessage.text("已结束对话").finish(reply_to=self.message_id)
        elif resp == "rollback":
            await self._handle_rollback()
        elif resp and isinstance(resp, str):
            self.context.append({"role": "user", "content": resp})

    async def _handle_rollback(self, steps: int = 1, by_error: bool = False) -> None:
        rollback_per_step = 1 if by_error else 2
        required_length = steps * rollback_per_step
        rollback_position = -rollback_per_step * steps

        if len(self.context) >= required_length:
            self.context = self.context[:rollback_position]
            action_desc = f"回滚 {steps} 条输入" if by_error else f"回滚 {steps} 轮对话"
            status_msg = f"Oops! 连接异常，已自动{action_desc}。" if by_error else f"已{action_desc}。"

            remaining_context = (
                "空" if not self.context else f"{self.context[-1]['role']}: {self.context[-1]['content']}"
            )

            await UniMessage.text(f"{status_msg}当前上下文为:\n{remaining_context}\nuser:（等待输入）").send(
                reply_to=self.message_id
            )
        elif by_error and len(self.context) > 0:
            self.context.clear()
            await self._message_reaction("fail")
            await UniMessage.text("Oops! 连接异常，请重新输入").send(reply_to=self.message_id)
        else:
            await UniMessage.text("无法回滚，当前对话记录为空").send(reply_to=self.message_id)

    async def _handle_tool_calls(self, message: Message) -> bool:
        if not message.tool_calls:
            return False

        try:
            result = await registry.execute_tool_call(message.tool_calls[0])
        except Exception:
            self.context.pop()
            return False

        self.context.append(
            {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "content": result,
            }
        )
        return True

    async def _get_response_message(self) -> Optional[Message]:
        try:
            completion = await API.chat(self.context, self.model.name)
            return completion.choices[0].message
        except (httpx.ReadTimeout, httpx.RequestError):
            if not self.is_contextual:
                await self._message_reaction("fail")
                await UniMessage.text("Oops! 网络超时，请稍后重试").finish(reply_to=self.message_id)
            await self._handle_rollback(by_error=True)
        except RequestException as e:
            if not self.is_contextual:
                await UniMessage.text(str(e)).finish(reply_to=self.message_id)
            await self._handle_rollback(by_error=True)

    def _extract_content_and_think(self, message: Message) -> tuple[str, str]:
        thinking = message.reasoning_content

        if not thinking:
            think_blocks = re.findall(r"<think>(.*?)</think>", message.content or "", flags=re.DOTALL)
            thinking = "\n".join([block.strip() for block in think_blocks if block.strip()])

        content = re.sub(r"<think>.*?</think>", "", message.content or "", flags=re.DOTALL).strip()

        return content, thinking

    def _format_output(self, message: Message, with_thinking: bool) -> str:
        content, thinking = self._extract_content_and_think(message)

        if with_thinking and content and thinking:
            return (
                f"<blockquote><p>{thinking}</p></blockquote>{content}"
                if self.is_to_pic
                else f"{thinking}\n\n--------------------\n\n{content}"
            )
        return content

    async def _send_response(self, message: Message) -> None:
        output = self._format_output(message, config.enable_send_thinking)
        message.reasoning_content = None

        await self._message_reaction("done")

        if self.is_use_tts and self.tts_model:
            try:
                output = self._format_output(message, False)
                unimsg = UniMessage.audio(raw=await API.text_to_speach(output, self.tts_model.name))
                await unimsg.send()
            except RequestException as e:
                tts_logger("ERROR", f"TTS Response error: {e}, Use image or text instead")
                output = self._format_output(message, config.enable_send_thinking)
                unimsg = (
                    UniMessage.image(raw=await self.md_to_pic(output))
                    if self.is_to_pic and callable(self.md_to_pic)
                    else UniMessage(output)
                )
                await unimsg.send(reply_to=self.message_id)
        elif self.is_to_pic and callable(self.md_to_pic):
            unimsg = UniMessage.image(raw=await self.md_to_pic(output))
            await unimsg.send(reply_to=self.message_id)
        else:
            await UniMessage.text(output).send(reply_to=self.message_id)
