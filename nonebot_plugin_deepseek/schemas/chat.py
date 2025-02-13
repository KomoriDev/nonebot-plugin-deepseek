from dataclasses import dataclass
from typing_extensions import TypeAlias
from typing import Literal, Optional, cast

from .usage import Usage
from .message import Message
from .logprobs import Logprobs

FinishReasonType: TypeAlias = Literal["stop", "length", "content_filter", "tool_calls", "insufficient_system_resource"]


class Delta:
    content: Optional[str] = None
    "实际输出的内容"
    reasoning_content: Optional[str] = None
    "推理输出的内容"
    role: Literal["assistant"] = "assistant"
    "角色"

    def __init__(
        self,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        role: Literal["assistant"] = "assistant",
        **kwargs,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.role = role

    def update(
        self,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        role: Optional[Literal["assistant"]] = None,
        **kwargs,
    ):
        if content:
            if self.content is None:
                self.content = content
            else:
                self.content += content
        if reasoning_content:
            if self.reasoning_content is None:
                self.reasoning_content = reasoning_content
            else:
                self.reasoning_content += reasoning_content
        if role:
            self.role = role


class StreamChoice:
    finish_reason: Optional[FinishReasonType] = None
    """模型停止生成 token 的原因"""
    index: int
    """该 completion 在模型生成的 completion 的选择列表中的索引"""
    delta: Delta
    """流式返回的一个 completion 增量。"""
    logprobs: Optional[Logprobs] = None
    """该 choice 的对数概率信息"""

    def __init__(
        self,
        index: int,
        delta: dict,
        *,
        finish_reason: Optional[FinishReasonType] = None,
        logprobs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.index = index
        self.delta = Delta(**delta)
        self.finish_reason = finish_reason
        if isinstance(logprobs, dict):
            self.logprobs = Logprobs(**logprobs)

    def update(self, other: "StreamChoice"):
        self.delta.update(
            content=other.delta.content,
            reasoning_content=other.delta.reasoning_content,
            role=other.delta.role,
        )
        if other.finish_reason:
            self.finish_reason = other.finish_reason
        if other.logprobs:
            self.logprobs = other.logprobs
        return self


class StreamChoiceList:
    id: str
    """该对话的唯一标识符。"""
    created: int
    """创建聊天完成时的 Unix 时间戳（以秒为单位）"""
    model: str
    """生成该 completion 的模型名"""
    object: Literal["chat.completion", "chat.completion.chunk"]
    """对象的类型, 其值为 `chat.completion`。流式传输时, 其值为 `chat.completion.chunk`"""
    system_fingerprint: Optional[str] = None
    """该指纹代表模型运行的后端配置"""
    usage: Optional[Usage] = None
    """该对话补全请求的用量信息"""

    def __init__(
        self,
        id: str,
        created: int,
        model: str,
        object: Literal["chat.completion", "chat.completion.chunk"],
        choices: list[dict],
        system_fingerprint: Optional[str] = None,
        usage=None,
        **kwargs,
    ) -> None:
        self.choices = [StreamChoice(**i) for i in choices]
        self.choices_index = [i.index for i in self.choices]
        self.id = id
        self.created = created
        self.model = model
        self.object = object
        self.system_fingerprint = system_fingerprint
        if isinstance(usage, dict):
            self.usage = Usage(**usage)

    def __add__(self, other: "StreamChoiceList"):
        if self.system_fingerprint != other.system_fingerprint:
            self.system_fingerprint = other.system_fingerprint
        if self.usage != other.usage:
            self.usage = other.usage

        for other_choice in other.choices:
            if other_choice.index not in self.choices_index:
                self.choices.append(other_choice)
                self.choices_index.append(other_choice.index)
                continue
            for self_choice in self.choices:
                if self_choice.index == other_choice.index:
                    self_choice.update(other_choice)

        return self

    def transform(self):
        return ChatCompletions(
            self.id,
            [
                Choice(
                    finish_reason=cast(FinishReasonType, choice.finish_reason),
                    index=choice.index,
                    message=Message(
                        choice.delta.role,
                        content=choice.delta.content,
                        reasoning_content=choice.delta.reasoning_content,
                    ),
                    logprobs=choice.logprobs,
                )
                for choice in self.choices
            ],
            self.created,
            self.model,
            self.object,
            self.usage
            if self.usage
            else Usage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
            self.system_fingerprint,
        )


@dataclass
class Choice:
    """模型生成的 completion 的选择列表"""

    finish_reason: FinishReasonType
    """模型停止生成 token 的原因"""
    index: int
    """该 completion 在模型生成的 completion 的选择列表中的索引"""
    message: Message
    """模型生成的 completion 消息"""
    logprobs: Optional[Logprobs] = None
    """该 choice 的对数概率信息"""

    def __post_init__(self) -> None:
        if isinstance(self.message, dict):
            self.message = Message(**self.message)
        if isinstance(self.logprobs, dict):
            self.logprobs = Logprobs(**self.logprobs)


@dataclass
class ChatCompletions:
    id: str
    """该对话的唯一标识符。"""
    choices: list[Choice]
    """模型生成的 completion 的选择列表"""
    created: int
    """创建聊天完成时的 Unix 时间戳（以秒为单位）"""
    model: str
    """生成该 completion 的模型名"""
    object: Literal["chat.completion", "chat.completion.chunk"]
    """对象的类型, 其值为 `chat.completion`。流式传输时, 其值为 `chat.completion.chunk`"""
    usage: Usage
    """该对话补全请求的用量信息"""
    system_fingerprint: Optional[str] = None
    """该指纹代表模型运行的后端配置"""

    def __post_init__(self) -> None:
        self.choices = [Choice(**choice) if isinstance(choice, dict) else choice for choice in self.choices]
        if isinstance(self.usage, dict):
            self.usage = Usage(**self.usage)
