import json
from pathlib import Path
from typing import Any, Union, Literal, Optional

from nonebot import get_plugin_config
from nonebot.compat import PYDANTIC_V2
import nonebot_plugin_localstore as store
from pydantic import Field, BaseModel, ConfigDict

from .log import ds_logger, tts_logger
from .exception import RequestException
from ._types import NOT_GIVEN, NotGivenOr
from .compat import model_dump, model_validator


class ModelConfig:
    def __init__(self) -> None:
        self.file: Path = store.get_plugin_config_dir() / "config.json"
        self.default_model: str = config.get_enable_models()[0]
        self.enable_md_to_pic: bool = config.md_to_pic
        self.tts_model_dict: dict[str, list[str]] = {}
        self.available_tts_models: list[str] = []
        self.default_tts_model: Optional[str] = None

        self.load()

    def load(self):
        if not self.file.exists():
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.save()
            return

        with open(self.file, encoding="utf-8") as f:
            data = json.load(f)
            self.default_model = data.get("default_model", self.default_model)
            self.enable_md_to_pic = data.get("enable_md_to_pic", self.enable_md_to_pic)
            self.default_tts_model = data.get("default_tts_model")
            if isinstance(data.get("available_tts_models"), dict):
                self.tts_model_dict = data.get("available_tts_models")
                self.available_tts_models = [
                    f"{model}-{spk}" for model, speakers in self.tts_model_dict.items() for spk in speakers
                ] + (tts_config.get_enable_tts() if tts_config.enable_tts_models else [])

        enable_models = config.get_enable_models()
        if self.default_model not in enable_models:
            self.default_model = enable_models[0]
            self.save()
        if self.enable_md_to_pic != config.md_to_pic:
            self.enable_md_to_pic = config.md_to_pic
            self.save()
        if self.available_tts_models and self.default_tts_model not in self.available_tts_models:
            self.default_tts_model = self.available_tts_models[0]
            self.save()
        if not self.available_tts_models and self.default_tts_model:
            self.save()

    def save(self):
        config_data = {
            "default_model": self.default_model,
            "enable_md_to_pic": self.enable_md_to_pic,
        }
        if self.default_tts_model in self.available_tts_models:
            config_data["default_tts_model"] = self.default_tts_model
        if self.available_tts_models:
            config_data["available_tts_models"] = self.tts_model_dict
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        self.load()


class CustomModel(BaseModel):
    name: str
    """Model Name"""
    base_url: str = "https://api.deepseek.com"
    """Custom base URL for this model (optional)"""
    alias: Optional[str] = None
    """Model alias name"""
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
                    ds_logger("WARNING", "不建议同时修改 `temperature` 和 `top_p` 字段")

                top_logprobs = data.get("top_logprobs")
                logprobs = data.get("logprobs")
                if top_logprobs and logprobs is False:
                    raise ValueError("指定 `top_logprobs` 参数时，`logprobs` 必须为 True")

            elif name == "deepseek-reasoner":
                max_tokens = data.get("max_tokens")
                if max_tokens and max_tokens > 8000:
                    ds_logger("WARNING", f"模型 {name} `max_tokens` 字段最大为 8000")

                unsupported_params = ["temperature", "top_p", "presence_penalty", "frequency_penalty"]
                params_present = [param for param in unsupported_params if param in data]
                if params_present:
                    ds_logger("WARNING", f"模型 {name} 不支持设置 {', '.join(params_present)}")

                logprobs = data.get("logprobs")
                top_logprobs = data.get("top_logprobs")
                if logprobs or top_logprobs:
                    raise ValueError(f"模型 {name} 不支持设置 logprobs、top_logprobs")

        return data

    def to_dict(self):
        return model_dump(
            self, exclude_unset=True, exclude_none=True, exclude={"name", "base_url", "alias", "api_key", "prompt"}
        )


class CustomTTS(BaseModel):
    name: str
    """TTS Preset Parameters Name"""
    model_name: str
    """TTS Model Name"""
    speaker_name: str
    """TTS Speaker Name"""
    prompt_text_lang: str = "中文"
    """language of the prompt text for the reference audio"""
    emotion: str = "随机"
    """Emotion"""
    text_lang: str = "中文"
    """language of the text to be synthesized"""
    top_k: int = Field(default=10, ge=1, le=100)
    """top k sampling"""
    top_p: Union[int, float] = Field(default=1, ge=0.01, le=1)
    """top p sampling"""
    temperature: Union[int, float] = Field(default=1, ge=0.01, le=1)
    """temperature for sampling"""
    text_split_method: str = "按标点符号切"
    """Text Split Method"""
    batch_size: int = Field(default=10, gt=1, lt=200)
    """ batch size for inference"""
    batch_threshold: Union[int, float] = Field(default=0.75, ge=0, le=1)
    """threshold for batch splitting."""
    split_bucket: bool = True
    """whether to split the batch into multiple buckets."""
    speed_facter: Union[int, float] = Field(default=1, ge=0.01, le=2)
    """control the speed of the synthesized audio."""
    fragment_interval: Union[int, float] = Field(default=0.3, ge=0.01, le=1)
    """Fragment Interval"""
    media_type: Literal["wav", "ogg", "acc"] = "wav"
    """Media Output Type"""
    parallel_infer: bool = True
    """Parallel Infer"""
    repetition_penalty: Union[int, float] = Field(default=1.35, ge=0, le=2)
    """repetition penalty for T2S model."""
    seed: int = -1
    """random seed for reproducibility."""

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    else:

        class Config:
            extra = "allow"
            arbitrary_types_allowed = True

    def to_dict(self):
        return model_dump(self, exclude_none=True, exclude={"name", "model_name", "speaker_name"})


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
        return [model.alias if model.alias else model.name for model in self.enable_models]

    def get_model_url(self, model_name: str) -> str:
        """Get the base_url corresponding to the model"""
        for model in self.enable_models:
            if model.name == model_name:
                return model.base_url
        raise ValueError(f"Model {model_name} not enabled")

    def get_model_config(self, model_name: str) -> CustomModel:
        """Get model config"""
        for model in self.enable_models:
            if model.name == model_name or model.alias == model_name:
                return model
        raise ValueError(f"Model {model_name} not enabled")


class ScopedTTSConfig(BaseModel):
    enable_tts_models: Union[list[CustomTTS], bool] = False
    """List of TTS models configurations"""
    base_url: str = ""
    """Your GPT-Sovits API Url """
    access_token: str = ""
    """Your GPT-Sovits API Access Token"""
    audio_dl_url: str = ""
    """audio download url"""

    @model_validator(mode="before")
    @classmethod
    def check_audio_dl_url(cls, data: dict) -> dict:
        if not data.get("audio_dl_url") and data.get("base_url"):
            data["audio_dl_url"] = data["base_url"]
        return data

    def get_enable_tts(self) -> list[str]:
        if isinstance(self.enable_tts_models, bool):
            return []
        return [model.name for model in self.enable_tts_models]

    async def get_available_tts(self) -> dict[str, list[str]]:
        from .apis import API

        try:
            tts_models = await API.get_tts_models()
            preset_dict = {model.model: list(model.speakers) for model in tts_models}
        except RequestException as e:
            preset_dict = {}
            tts_logger("WARNING", f"获取 TTS 模型列表失败: {e}")
        return preset_dict

    def get_tts_model(self, preset_name: str) -> CustomTTS:
        """Get TTS model config"""
        if not isinstance(self.enable_tts_models, bool):
            for model in self.enable_tts_models:
                if (
                    model.name == preset_name
                    and f"{model.model_name}-{model.speaker_name}" in model_config.available_tts_models
                ):
                    return model
        if "-" in preset_name:
            model_name = preset_name.split("-")[0]
            speaker_name = preset_name.split("-")[1]
            if preset_name in model_config.available_tts_models:
                return CustomTTS(name=preset_name, model_name=model_name, speaker_name=speaker_name)
        raise ValueError(f"TTS Model {preset_name} not valid")


class Config(BaseModel):
    deepseek: ScopedConfig = Field(default_factory=ScopedConfig)
    """DeepSeek Plugin Config"""
    deepseek_tts: ScopedTTSConfig = Field(default_factory=ScopedTTSConfig)
    """DeepSeek TTS Plugin Config"""


config = (get_plugin_config(Config)).deepseek
tts_config = (get_plugin_config(Config)).deepseek_tts
model_config = ModelConfig()
ds_logger("DEBUG", f"load deepseek model: {config.get_enable_models()}")
