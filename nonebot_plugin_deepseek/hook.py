from nonebot import get_driver
from nonebot_plugin_alconna import command_manager
from nonebot_plugin_localstore import get_plugin_cache_dir

from .log import ds_logger

driver = get_driver()
cach_dir = get_plugin_cache_dir() / "shortcut.db"


@driver.on_startup
async def _() -> None:
    command_manager.load_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache loaded")


@driver.on_shutdown
async def _() -> None:
    command_manager.dump_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache dumped")
