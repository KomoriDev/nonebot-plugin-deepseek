from importlib.metadata import metadata

from nonebot.compat import PYDANTIC_V2
from pydantic import BaseModel, ConfigDict


class Metadata(BaseModel):
    name: str
    version: str
    summary: str

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:

        class Config:
            extra = "allow"


__metadata__ = Metadata(**metadata("nonebot-plugin-deepseek").json)  # type: ignore

__version__ = __metadata__.version
