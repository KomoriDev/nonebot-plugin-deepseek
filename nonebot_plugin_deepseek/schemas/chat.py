from dataclasses import dataclass
from typing import Literal, Optional
from typing_extensions import TypeAlias

from .usage import Usage
from .message import Message
from .logprobs import Logprobs

FinishReasonType: TypeAlias = Literal["stop", "length", "content_filter", "tool_calls", "insufficient_system_resource"]


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
    object: Literal["chat.completion"]
    """对象的类型, 其值为 `chat.completion`"""
    usage: Usage
    """该对话补全请求的用量信息"""
    system_fingerprint: Optional[str] = None
    """该指纹代表模型运行的后端配置"""

    def __post_init__(self) -> None:
        self.choices = [Choice(**choice) if isinstance(choice, dict) else choice for choice in self.choices]
        if isinstance(self.usage, dict):
            self.usage = Usage(**self.usage)
