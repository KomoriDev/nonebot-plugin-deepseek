from clilte import CommandLine
from arclet.alconna import Argv, set_default_argv_type

from .plugins.tts import TTSUpdate

set_default_argv_type(Argv)
deepseek = CommandLine(
    "NB CLI plugin for nonebot-plugin-deepseek",
    "0.1.8",
    rich=True,
    _name="nb deepseek",
    load_preset=True,
)
deepseek.add(TTSUpdate)
