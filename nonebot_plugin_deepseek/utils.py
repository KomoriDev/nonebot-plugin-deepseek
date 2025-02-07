import re
from dataclasses import asdict
from typing import Any, Union, Literal, Optional

import httpx
from nonebot.adapters import Event
from nonebot.permission import User, Permission
from nonebot_plugin_htmlrender import md_to_pic
from nonebot_plugin_waiter import Waiter, prompt
from nonebot_plugin_alconna.uniseg import UniMsg, UniMessage
from nonebot.matcher import Matcher, current_event, current_matcher

from .apis import API
from .schemas import Message
from .config import CustomModel, config
from .function_call.registry import registry


class DeepSeekHandler:
    def __init__(
        self,
        model: CustomModel,
        is_to_pic: bool,
        is_contextual: bool,
    ) -> None:
        self.model: CustomModel = model
        self.is_to_pic: bool = is_to_pic
        self.is_contextual: bool = is_contextual

        self.event: Event = current_event.get()
        self.matcher: Matcher = current_matcher.get()
        self.message_id: str = UniMessage.get_message_id(self.event)
        self.waiter: Waiter[Union[str, Literal[False]]] = self._setup_waiter()

        self.context: list[dict[str, Any]] = []

    async def handle(self, content: Optional[str]) -> None:
        if content:
            self.context.append({"role": "user", "content": content})

        if not self.is_contextual:
            await self._handle_single_conversion()
        else:
            await self._handle_multi_round_conversion()

    async def _handle_single_conversion(self) -> None:
        if message := await self._get_response_message():
            await self._send_response(message)

    async def _handle_multi_round_conversion(self) -> None:
        async for resp in self.waiter(default=False, timeout=config.context_timeout):
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
        waiter = Waiter(waits=["message"], handler=self._waiter_handler, matcher=self.matcher, permission=permission)
        waiter.future.set_result("")
        return waiter

    def _waiter_handler(self, msg: UniMsg, skip: bool = False) -> Union[str, Literal[False]]:
        text = msg.extract_plain_text()
        if not skip:
            self.message_id = msg.get_message_id()
        if text in ["结束", "取消", "done"]:
            return False
        if text in ["回滚", "rollback"]:
            return "rollback"
        return text

    def _prompt_handler(self, msg: UniMsg) -> UniMsg:
        self.message_id = msg.get_message_id()
        return msg

    async def _process_waiter_response(self, resp: Union[bool, str]) -> None:
        if resp == "" and not self.context:
            _resp = await prompt("你想对 DeepSeek 说什么呢？", handler=self._prompt_handler, timeout=60)
            if _resp is None:
                await UniMessage("等待超时").finish(reply_to=self.message_id)
            resp = self._waiter_handler(_resp, skip=True)

        if resp is False:
            await UniMessage("已结束对话").finish(reply_to=self.message_id)
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
                "空" if not self.context else f'{self.context[-1]["role"]}: {self.context[-1]["content"]}'
            )

            await UniMessage.text(f"{status_msg}当前上下文为:\n{remaining_context}\n" "user:（等待输入）").send(
                reply_to=self.message_id
            )
        elif by_error and len(self.context) > 0:
            self.context.clear()
            await UniMessage("Oops! 连接异常，请重新输入").send(reply_to=self.message_id)
        else:
            await UniMessage("无法回滚，当前对话记录为空").send(reply_to=self.message_id)

    async def _handle_tool_calls(self, message: Message) -> bool:
        if not message.tool_calls:
            return False

        try:
            result = await registry.execute_tool_call(message.tool_calls[0])
        except Exception:
            self.context.pop()
            return False

        self.context.append({"role": "tool", "tool_call_id": message.tool_calls[0].id, "content": result})
        return True

    async def _get_response_message(self) -> Optional[Message]:
        try:
            completion = await API.chat(self.context, self.model.name)
            return completion.choices[0].message
        except (httpx.ReadTimeout, httpx.RequestError):
            if not self.is_contextual:
                await UniMessage("Oops! 网络超时，请稍后重试").finish(reply_to=self.message_id)
            await self._handle_rollback(by_error=True)

    def _extract_content_and_think(self, message: Message) -> tuple[str, str]:
        thinking = message.reasoning_content

        if not thinking:
            think_blocks = re.findall(r"<think>(.*?)</think>", message.content or "", flags=re.DOTALL)
            thinking = "\n".join([block.strip() for block in think_blocks if block.strip()])

        content = re.sub(r"<think>.*?</think>", "", message.content or "", flags=re.DOTALL).strip()

        return content, thinking

    def _format_output(self, message: Message) -> str:
        content, thinking = self._extract_content_and_think(message)

        if config.enable_send_thinking and content and thinking:
            return (
                f"<blockquote><p>{thinking}</p></blockquote>{content}"
                if self.is_to_pic
                else f"{thinking}\n----\n{content}"
            )
        return content

    async def _send_response(self, message: Message) -> None:
        output = self._format_output(message)
        message.reasoning_content = None
        if self.is_to_pic:
            if unimsg := UniMessage.image(raw=await md_to_pic(output)):
                await unimsg.send(reply_to=self.message_id)
        else:
            await UniMessage(output).send(reply_to=self.message_id)
