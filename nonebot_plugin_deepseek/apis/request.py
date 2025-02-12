import httpx
from nonebot.log import logger

from nonebot_plugin_deepseek.schemas.tts import TTSResponse

from ..compat import model_dump
from ..config import config, tts_config

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions


class API:
    _headers = {
        "Accept": "application/json",
    }

    @classmethod
    async def chat(cls, message: list[dict[str, str]], model: str = "deepseek-chat") -> ChatCompletions:
        """普通对话"""
        model_config = config.get_model_config(model)

        api_key = model_config.api_key or config.api_key
        prompt = model_dump(model_config, exclude_none=True).get("prompt", config.prompt)

        json = {
            "messages": [{"content": prompt, "role": "system"}] + message if prompt else message,
            "model": model,
            **model_config.to_dict(),
        }
        logger.debug(f"使用模型 {model}，配置：{json}")
        # if model == "deepseek-chat":
        #     json.update({"tools": registry.to_json()})
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{model_config.base_url}/chat/completions",
                headers={**cls._headers, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=json,
                timeout=50,
            )
        if error := response.json().get("error"):
            raise RequestException(error["message"])
        return ChatCompletions(**response.json())

    @classmethod
    async def query_balance(cls, model_name: str) -> Balance:
        model_config = config.get_model_config(model_name)
        api_key = model_config.api_key or config.api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{model_config.base_url}/user/balance",
                headers={**cls._headers, "Authorization": f"Bearer {api_key}"},
            )
        if response.status_code == 404:
            raise RequestException("本地模型不支持查询余额，请更换默认模型")
        return Balance(**response.json())

    @classmethod
    async def get_tts_models(cls) -> list[TTSResponse]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{tts_config.base_url}/models",
                headers={**cls._headers},
                timeout=30,
            )
        if response.status_code != 200:
            raise RequestException("获取 TTS 模型列表失败")
        return [await TTSResponse.create(model=model) for model in response.json()]

    @classmethod
    async def get_tts_speakers(cls, model_name: str) -> list[str]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tts_config.base_url}/spks",
                headers={**cls._headers},
                json={"model": model_name},
                timeout=30,
            )
        if speakers := response.json().get("speakers"):
            return list(speakers.keys())
        else:
            raise RequestException("获取 TTS 模型讲话人列表失败")

    @classmethod
    async def text_to_speach(cls, text: str, model: str) -> bytes:
        model_config = tts_config.get_tts_model(model)
        model_name = model_config.model_name
        speaker = model_config.speaker_name
        json = {
            "text": text,
            "model_name": model_name,
            "speaker_name": speaker,
            "app_key": tts_config.access_token,
            "access_token": tts_config.access_token,
            "audio_dl_url": tts_config.audio_dl_url,
            **model_config.to_dict(),
        }

        logger.debug(f"[GPT-Sovits] 使用模型 {model}，讲话人：{speaker}, 配置：{json}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tts_config.base_url}/infer_single",
                headers={**cls._headers},
                json=json,
                timeout=50,
            )
        logger.debug(f"[GPT-Sovits] Response: {response.text}")
        if audio_url := response.json().get("audio_url"):
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url)
                return response.content
        else:
            raise RequestException("语音合成失败")
