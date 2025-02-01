from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Function:
    """模型调用的 function"""

    name: str
    """模型调用的 function 名"""
    arguments: str
    """要调用的 function 的参数，由模型生成，格式为 JSON。"""


@dataclass
class ToolCalls:
    """模型生成的 tool 调用，例如 function 调用。"""

    index: int
    id: str
    """tool 调用的 ID"""
    type: Literal["function"]
    """tool 的类型。目前仅支持 `function`"""
    function: Function
    """模型调用的 function"""

    def __post_init__(self) -> None:
        if isinstance(self.function, dict):
            self.function = Function(**self.function)


@dataclass
class Message:
    """模型生成的 completion 消息"""

    role: Literal["assistant"]
    """生成这条消息的角色"""
    content: Optional[str] = None
    """该 completion 的内容"""
    reasoning_content: Optional[str] = None
    """
    仅适用于 deepseek-reasoner 模型。内容为 assistant 消息中在最终答案之前的推理内容
    """
    tool_calls: Optional[list[ToolCalls]] = None
    """模型生成的 tool 调用"""

    def __post_init__(self) -> None:
        if self.tool_calls:
            self.tool_calls = [
                ToolCalls(**tool_call) if isinstance(tool_call, dict) else tool_call for tool_call in self.tool_calls
            ]
