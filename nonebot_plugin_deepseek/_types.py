"""
@author: openai
@website: https://github.com/openai/openai-python/blob/main/src/openai/_types.py
"""

from typing import Literal, TypeVar, override

_T = TypeVar("_T")


class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: int | NotGiven | None = NotGiven()) -> Response: ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = _T | NotGiven
NOT_GIVEN = NotGiven()
