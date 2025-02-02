from typing import Literal
from dataclasses import dataclass


@dataclass
class BalanceInfo:
    currency: Literal["CNY", "USD"]
    """货币，人民币或美元"""
    total_balance: str
    """总的可用余额，包括赠金和充值余额"""
    granted_balance: str
    """未过期的赠金余额"""
    topped_up_balance: str
    """充值余额"""


@dataclass
class Balance:
    """用户余额详情"""

    is_available: bool
    """当前账户是否有余额可供 API 调用"""
    balance_infos: list[BalanceInfo]

    def __post_init__(self) -> None:
        self.balance_infos = [BalanceInfo(**info) if isinstance(info, dict) else info for info in self.balance_infos]
