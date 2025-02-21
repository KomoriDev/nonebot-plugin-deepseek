from nonebot import get_driver
from nonebot_plugin_localstore import get_plugin_cache_dir
from nonebot_plugin_alconna import Args, Field, Option, Subcommand, command_manager

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
            tts_logger("DEBUG", f"First load, querying available TTS models: {available_models}")
        else:
            tts_logger("DEBUG", f"load available tts models: {model_config.available_tts_models}")
        command = command_manager.get_command("deepseek::deepseek")
        command.add(
            Subcommand(
                "tts",
                Option("-l|--list", help_text="支持的TTS模型列表"),
                Option(
                    "--set-default",
                    Args[
                        "model#模型名称",
                        model_config.available_tts_models,
                        Field(
                            completion=lambda: f"请输入TTS模型预设名，预期为："
                            f"{model_config.available_tts_models[:10]}…… 其中之一\n"
                            "输入 `/deepseek tts -l` 查看所有TTS模型及角色"
                        ),
                    ],
                    dest="set",
                    help_text="设置默认TTS模型",
                ),
                help_text="TTS模型相关设置",
            ),
        )
        command.add(Option("--use-tts", help_text="使用TTS回复"))
        command.shortcut("TTS模型列表", {"command": "deepseek tts --list", "fuzzy": False, "prefix": True})
        command.shortcut("设置默认TTS模型", {"command": "deepseek tts --set-default", "fuzzy": True, "prefix": True})
        command.shortcut(
            "多轮语音对话", {"command": "deepseek --use-tts --with-context", "fuzzy": True, "prefix": True}
        )
        tts_logger("DEBUG", "Loaded TTS Subcommands")
    command_manager.load_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache loaded")


@driver.on_shutdown
async def _() -> None:
    command_manager.dump_cache(cach_dir)
    ds_logger("DEBUG", "DeepSeek shortcuts cache dumped")
