import json
from pathlib import Path
from typing_extensions import Self
from typing import Any, Union, Optional

from nonebot.compat import PYDANTIC_V2
import nonebot_plugin_localstore as store
from nonebot import logger, get_plugin_config
from pydantic import Field, BaseModel, ConfigDict, model_validator


class ModelConfig:
    def __init__(self) -> None:
        self.file: Path = store.get_plugin_config_dir() / "config.json"
        self.default_model: str = config.get_enable_models()[0]
        self.default_prompt: str = config.prompt  # 暂时用不到
        self.load()

    def load(self):
        if not self.file.exists():
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.save()
            return

        with open(self.file) as f:
            data = json.load(f)
            self.default_model = data.get("default_model", self.default_model)
            self.default_prompt = data.get("default_prompt", self.default_prompt)

    def save(self):
        config_data = {
            "default_model": self.default_model,
            "default_prompt": self.default_prompt,
        }
        with open(self.file, "w") as f:
            json.dump(config_data, f, indent=2)
        self.load()


class CustomModel(BaseModel):
    name: str
    """Model Name"""
    base_url: str = "https://api.deepseek.com"
    """Custom base URL for this model (optional)"""
    max_tokens: int = Field(default=4090, gt=1, lt=8192)
    """
    限制一次请求中模型生成 completion 的最大 token 数
    - `deepseek-chat`: Integer between 1 and 8192. Default is 4090.
    - `deepseek-reasoner`: Default is 4K, maximum is 8K.
    """
    frequency_penalty: Union[int, float] = Field(default=0, ge=-2, le=2)
    """
    Discourage the model from repeating the same words or phrases too frequently within the generated text
    """
    presence_penalty: Union[int, float] = Field(default=0, ge=-2, le=2)
    """Encourage the model to include a diverse range of tokens in the generated text"""
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    """
    Stop generating tokens when encounter these words.
    Note that the list contains a maximum of 16 string.
    """
    temperature: Union[int, float] = Field(default=1, ge=0, le=2)
    """Sampling temperature. It is not recommended to used it with top_p"""
    top_p: Union[int, float] = Field(default=1, ge=0, le=1)
    """Alternatives to sampling temperature. It is not recommended to used it with temperature"""
    logprobs: Optional[bool] = Field(default=None)
    """Whether to return the log probability of the output token."""
    top_logprobs: Optional[int] = Field(default=None, le=20)
    """Specifies that the most likely token be returned at each token position."""

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:

        class Config:
            extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def check_max_token(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "max_tokens" not in data:
                name = data.get("name")
                if name == "deepseek-reasoner":
                    data["max_tokens"] = 4000
                else:
                    data["max_tokens"] = 4090
        return data

    @model_validator(mode="after")
    def check_model(self) -> Self:
        if self.stop and isinstance(self.stop, list):
            if len(self.stop) >= 16:
                raise ValueError("字段 `stop` 最多允许设置 16 个字符")

        if self.name == "deepseek-chat":
            if self.temperature and self.top_p:
                logger.warning("不建议同时修改 `temperature` 和 `top_p` 字段")
            if self.top_logprobs and (not self.logprobs or self.logprobs is False):
                raise ValueError("指定 `top_logprobs` 参数时，`logprobs` 必须为 True")

        if self.name == "deepseek-reasoner":
            if self.max_tokens > 8000:
                logger.warning(f"模型 {self.name} `max_tokens` 字段最大为 8000")
            if self.temperature or self.top_p or self.presence_penalty or self.frequency_penalty:
                logger.warning(f"模型 {self.name} 不支持设置 temperature、top_p、presence_penalty、frequency_penalty")
            if self.logprobs or self.top_logprobs:
                raise ValueError(f"模型 {self.name} 不支持设置 logprobs、top_logprobs")

        return self


class ScopedConfig(BaseModel):
    api_key: str = ""
    """Your API Key from deepseek"""
    enable_models: list[CustomModel] = [
        CustomModel(name="deepseek-chat"),
        CustomModel(name="deepseek-reasoner"),
    ]
    """List of models configurations"""
    prompt: str = ""
    """Character Preset"""
    md_to_pic: bool = False
    """Text to Image"""
    enable_send_thinking: bool = False
    """Whether to send model thinking chain"""

    def get_enable_models(self) -> list[str]:
        return [model.name for model in self.enable_models]

    def get_model_url(self, model_name: str) -> str:
        """Get the base_url corresponding to the model"""
        for model in self.enable_models:
            if model.name == model_name:
                return model.base_url
        raise ValueError(f"Model {model_name} not enabled")


class Config(BaseModel):
    deepseek: ScopedConfig = Field(default_factory=ScopedConfig)
    """DeepSeek Plugin Config"""


config = (get_plugin_config(Config)).deepseek
model_config = ModelConfig()
logger.debug(f"load deepseek model: {config.get_enable_models()}")
