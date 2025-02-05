from typing import Literal, overload

from nonebot.compat import PYDANTIC_V2
from nonebot.compat import model_dump as model_dump

__all__ = (
    "model_dump",
    "model_validator",
)


if PYDANTIC_V2:
    from pydantic import model_validator as model_validator
else:
    from pydantic import root_validator

    @overload
    def model_validator(*, mode: Literal["before"]): ...

    @overload
    def model_validator(*, mode: Literal["after"]): ...

    def model_validator(*, mode: Literal["before", "after"]):
        return root_validator(pre=mode == "before", allow_reuse=True)  # type: ignore
