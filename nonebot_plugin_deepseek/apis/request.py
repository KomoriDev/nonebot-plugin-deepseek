import httpx

from ..config import config
from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions


class API:
    _headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    @classmethod
    async def chat(cls, message: list[dict[str, str]]) -> ChatCompletions:
        """普通对话"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.base_url}/chat/completions",
                headers={**cls._headers, "Content-Type": "application/json"},
                json={
                    "messages": [
                        {"content": config.prompt, "role": "system"},
                    ]
                    + message,
                    "model": config.model,
                    "response_format": {"type": "text"},
                    "stop": None,
                    "stream": False,
                    "tools": registry.to_json(),
                },
                timeout=config.timeout,
            )
        if error := response.json().get("error"):
            raise RequestException(error["message"])
        return ChatCompletions(**response.json())

    @classmethod
    async def query_balance(cls) -> Balance:
        """查询账号余额"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.base_url}/user/balance", headers=cls._headers
            )

        return Balance(**response.json())
