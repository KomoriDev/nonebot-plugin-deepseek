import json
from pathlib import Path
from typing import Any, Union, Optional

from nonebot.compat import PYDANTIC_V2
import nonebot_plugin_localstore as store
from nonebot import logger, get_plugin_config
from pydantic import Field, BaseModel, ConfigDict

from ._types import NOT_GIVEN, NotGivenOr
from .compat import model_dump, model_validator


class ModelConfig:
    def __init__(self) -> None:
        self.file: Path = store.get_plugin_config_dir() / "config.json"
        self.default_model: str = config.get_enable_models()[0]
        self.enable_md_to_pic: bool = config.md_to_pic
        self.load()

    def load(self):
        if not self.file.exists():
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.save()
            return

        with open(self.file) as f:
            data = json.load(f)
            self.default_model = data.get("default_model", self.default_model)
            self.enable_md_to_pic = data.get("enable_md_to_pic", self.enable_md_to_pic)

        enable_models = config.get_enable_models()
        if self.default_model not in enable_models:
            self.default_model = enable_models[0]
            self.save()

    def save(self):
        config_data = {
            "default_model": self.default_model,
            "enable_md_to_pic": self.enable_md_to_pic,
        }
        with open(self.file, "w") as f:
            json.dump(config_data, f, indent=2)
        self.load()


class CustomModel(BaseModel):
    name: str
    """Model Name"""
    base_url: str = "https://api.deepseek.com"
    """Custom base URL for this model (optional)"""
    api_key: Optional[str] = None
    """Custom API Key for the model (optional)"""
    prompt: Optional[str] = None
    """Custom character preset for the model (optional)"""
    stream: Optional[bool] = Field(default=None)
    """Streaming"""
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
    logprobs: NotGivenOr[Union[bool, None]] = Field(default=NOT_GIVEN)
    """Whether to return the log probability of the output token."""
    top_logprobs: NotGivenOr[int] = Field(default=NOT_GIVEN, le=20)
    """Specifies that the most likely token be returned at each token position."""

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    else:

        class Config:
            extra = "allow"
            arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def check_max_token(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = data.get("name")

            if "max_tokens" not in data:
                if name == "deepseek-reasoner":
                    data["max_tokens"] = 4000
                else:
                    data["max_tokens"] = 4090

            stop = data.get("stop")
            if isinstance(stop, list) and len(stop) >= 16:
                raise ValueError("字段 `stop` 最多允许设置 16 个字符")

            if name == "deepseek-chat":
                temperature = data.get("temperature")
                top_p = data.get("top_p")
                if temperature and top_p:
                    logger.warning("不建议同时修改 `temperature` 和 `top_p` 字段")

                top_logprobs = data.get("top_logprobs")
                logprobs = data.get("logprobs")
                if top_logprobs and logprobs is False:
                    raise ValueError("指定 `top_logprobs` 参数时，`logprobs` 必须为 True")

            elif name == "deepseek-reasoner":
                max_tokens = data.get("max_tokens")
                if max_tokens and max_tokens > 8000:
                    logger.warning(f"模型 {name} `max_tokens` 字段最大为 8000")

                unsupported_params = ["temperature", "top_p", "presence_penalty", "frequency_penalty"]
                params_present = [param for param in unsupported_params if param in data]
                if params_present:
                    logger.warning(f"模型 {name} 不支持设置 {', '.join(params_present)}")

                logprobs = data.get("logprobs")
                top_logprobs = data.get("top_logprobs")
                if logprobs or top_logprobs:
                    raise ValueError(f"模型 {name} 不支持设置 logprobs、top_logprobs")

        return data

    def to_dict(self):
        return model_dump(
            self, exclude_unset=True, exclude_none=True, exclude={"name", "base_url", "api_key", "prompt"}
        )


class TimeoutConfig(BaseModel):
    api_request: int = Field(default=100)
    """API request timeout (Not applicable for streaming)"""
    user_input: int = Field(default=60)
    """User input timeout"""


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
    timeout: Union[int, TimeoutConfig] = Field(default_factory=TimeoutConfig)
    """Timeout"""
    stream: bool = False
    """Stream"""

    def get_enable_models(self) -> list[str]:
        return [model.name for model in self.enable_models]

    def get_model_url(self, model_name: str) -> str:
        """Get the base_url corresponding to the model"""
        for model in self.enable_models:
            if model.name == model_name:
                return model.base_url
        raise ValueError(f"Model {model_name} not enabled")

    def get_model_config(self, model_name: str) -> CustomModel:
        """Get model config"""
        for model in self.enable_models:
            if model.name == model_name:
                return model
        raise ValueError(f"Model {model_name} not enabled")


class Config(BaseModel):
    deepseek: ScopedConfig = Field(default_factory=ScopedConfig)
    """DeepSeek Plugin Config"""


config = (get_plugin_config(Config)).deepseek
model_config = ModelConfig()
logger.debug(f"load deepseek model: {config.get_enable_models()}")
