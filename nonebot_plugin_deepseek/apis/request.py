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
