import json
import asyncio
from asyncio.queues import Queue

from httpx import AsyncClient
from nonebot.log import logger

from ..config import config

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatChunkedCompletions


class API:
    _client = AsyncClient()
    _client.headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    @classmethod
    async def chat(cls, message: list[dict[str, str]], model: str = "deepseek-chat") -> ChatChunkedCompletions:
        """普通对话"""
        chunk_queue = Queue()

        async def send_request():
            model_config = config.get_model_config(model)
            json = {
                "stream": True,
                "model": model,
                "messages": [{"content": config.prompt, "role": "user"}] + message if config.prompt else message,
                **model_config.to_dict(),
            }
            logger.debug(f"使用模型 {model}，配置：{json}")
            async with cls._client.stream("POST", f"https://api.deepseek.com/chat/completions", json=json) as response:
                async for chunk in response.aiter_lines():
                    await chunk_queue.put(chunk)

        # if model == "deepseek-chat":
        #     json.update({"tools": registry.to_json()})
        
        result_model = None
        result_message = ""

        request_task = asyncio.create_task(send_request())

        while True:
            chunk = await asyncio.wait_for(chunk_queue.get(), timeout=10)
            if not chunk:
                continue
            chunk = chunk.lstrip('data:').strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if error := data.get("error"):
                request_task.cancel()
                raise RequestException(error["message"])
            if data.get("usage"):
                data["choices"][0].pop("delta")
                data["choices"][0]["message"] = result_message
                result_model = ChatChunkedCompletions(**data)
                continue
            result_message += data["choices"][0]["delta"]["content"]
        if not request_task.done():
            request_task.cancel()
        return result_model

    @classmethod
    async def query_balance(cls) -> Balance:
        """查询账号余额"""
        response = await cls._client.get(f"{config.get_model_url('deepseek-chat')}/user/balance")
        return Balance(**response.json())
