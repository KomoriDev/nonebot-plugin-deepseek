import httpx
from nonebot.log import logger

from ..config import config

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions


class API:
    _headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    @classmethod
    async def chat(cls, message: list[dict[str, str]], model: str = "deepseek-chat") -> ChatCompletions:
        """普通对话"""
        model_config = config.get_model_config(model)

        """检测模型配置prompt"""
        prompt = config.prompt
        if model_config.prompt is not None:
            prompt = model_config.prompt
            logger.debug(f"使用模型内prompt {prompt}")
        else:
            logger.debug(f"使用全局prompt {prompt}")

        """检测模型配置api_key"""
        if model_config.api_key is not None:
            cls._headers["Authorization"] = f"Bearer {model_config.api_key}"
            logger.debug(f"使用模型内api_key {model_config.api_key}")
        else:
            cls._headers["Authorization"] = f"Bearer {config.api_key}"
            logger.debug(f"使用全局api_key {config.api_key}")

        json = {
            "messages": [{"content": prompt, "role": "system"}] + message
            if prompt #删除了对deepseek-chat的判断
            else message,
            "model": model,
            **model_config.to_dict(),
        }
        logger.debug(f"使用模型 {model}，配置：{json}")
        # if model == "deepseek-chat":
        #     json.update({"tools": registry.to_json()})
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{model_config.base_url}/chat/completions",
                headers={**cls._headers, "Content-Type": "application/json"},
                json=json,
                timeout=50,
            )
        if error := response.json().get("error"):
            raise RequestException(error["message"])
        return ChatCompletions(**response.json())

    @classmethod
    async def query_balance(cls) -> Balance:
        """查询账号余额"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.get_model_url('deepseek-chat')}/user/balance",
                headers=cls._headers,
            )

        return Balance(**response.json())
