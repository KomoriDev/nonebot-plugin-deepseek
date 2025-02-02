from typing import Optional
from dataclasses import dataclass


@dataclass
class TopLogprobs:
    """
    一个包含在该输出位置上，输出概率 top N 的 token 的列表，以及它们的对数概率
    在罕见情况下，返回的 token 数量可能少于请求参数中指定的 `top_logprobs` 值
    """

    token: str
    """输出的 token"""
    logprob: int
    """
    该 token 的对数概率
    `-9999.0` 代表该 token 的输出概率极小，不在 top 20 最可能输出的 token 中"""
    bytes: Optional[list[int]] = None
    """
    一个包含该 token UTF-8 字节表示的整数列表
    一般在一个 UTF-8 字符被拆分成多个 token 来表示时有用
    如果 token 没有对应的字节表示，则该值为 `None`
    """


@dataclass
class Content:
    """一个包含输出 token 对数概率信息的列表"""

    token: str
    """输出的 token"""
    logprob: int
    """
    该 token 的对数概率
    `-9999.0` 代表该 token 的输出概率极小，不在 top 20 最可能输出的 token 中"""
    top_logprobs: list[TopLogprobs]
    """
    一个包含在该输出位置上，输出概率 top N 的 token 的列表，以及它们的对数概率
    在罕见情况下，返回的 token 数量可能少于请求参数中指定的 `top_logprobs` 值
    """
    bytes: Optional[list[int]] = None
    """
    一个包含该 token UTF-8 字节表示的整数列表
    一般在一个 UTF-8 字符被拆分成多个 token 来表示时有用
    如果 token 没有对应的字节表示，则该值为 `None`
    """

    def __post_init__(self) -> None:
        if self.top_logprobs:
            self.top_logprobs = [
                TopLogprobs(**top_logprob) if isinstance(top_logprob, dict) else top_logprob
                for top_logprob in self.top_logprobs
            ]


@dataclass
class Logprobs:
    """该 choice 的对数概率信息"""

    content: Optional[list[Content]] = None

    def __post_init__(self) -> None:
        if self.content:
            self.content = [Content(**content) if isinstance(content, dict) else content for content in self.content]
