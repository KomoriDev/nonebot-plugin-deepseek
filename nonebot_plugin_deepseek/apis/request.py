import httpx
from nonebot.log import logger

from ..config import config
from ..compat import model_dump

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
    async def get_tts_models(cls) -> list[str]:
        # Todo: Test and implement this method. Not Finished.
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.enable_tts}/models",
                headers={**cls._headers},
            )
        return response.json()

    @classmethod
    async def get_tts_speakers(cls, model_name: str) -> list[str]:
        # Todo: Test and implement this method. Not Finished.
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.enable_tts}/spks",
                headers={**cls._headers},
                json={"model_name": model_name},
            )
        return response.json()

    @classmethod
    async def enable_tts(cls, text: str) -> bytes:
        json = {
            "app_key": "",
            "audio_dl_url": "",
            "model_name": config.default_tts_model,
            "speaker_name": config.default_tts_speaker,
            "prompt_text_lang": "中文",
            "emotion": "随机",
            "text": text,
            "text_lang": "中文",
            "top_k": 10,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "按标点符号切",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_facter": 1,
            "fragment_interval": 0.3,
            "media_type": "wav",
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "seed": -1,
        }
        logger.debug(
            f"[GPT-Sovits] 使用模型 {config.default_tts_model}，讲话人：{config.default_tts_speaker}, 配置：{json}"
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.tts_api_url}/infer_single",
                headers={**cls._headers},
                json=json,
                timeout=50,
            )
        if audio_url := response.json().get("audio_url"):
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url)
                return response.content
        else:
            raise RequestException("语音合成失败")
