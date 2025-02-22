from nonebot import get_driver
from nonebot_plugin_alconna import command_manager
from nonebot_plugin_localstore import get_plugin_cache_dir

from .log import ds_logger, tts_logger
from .config import tts_config, model_config

driver = get_driver()
cach_dir = get_plugin_cache_dir() / "shortcut.db"


@driver.on_startup
async def _() -> None:
    if tts_config.enable_tts_models:
        if not model_config.available_tts_models:
            available_models = await tts_config.get_available_tts()
            model_config.available_tts_models = [
                f"{model}-{spk}" for model, speakers in available_models.items() for spk in speakers
            ]
            model_config.tts_model_dict = available_models
            model_config.save()
            tts_logger("DEBUG", f"Loaded available TTS models: {available_models}")
        else:
            tts_logger("DEBUG", f"Loaded available TTS models: {model_config.available_tts_models}")
    command_manager.load_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache loaded")


@driver.on_shutdown
async def _() -> None:
    command_manager.dump_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache dumped")
