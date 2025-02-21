from json import loads
from typing import Union, Literal, Optional

import httpx
from nonebot.log import logger

from ..config import config
from ..compat import model_dump

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions, StreamChoiceList


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
            "messages": ([{"content": prompt, "role": "system"}] + message if prompt else message),
            "model": model,
            **model_config.to_dict(),
        }
        logger.debug(f"使用模型 {model}，配置：{json}")
        # if model == "deepseek-chat":
        #     json.update({"tools": registry.to_json()})
        if model_dump(model_config, exclude_none=True).get("stream", config.stream):
            ret = await stream_request(model_config.base_url, api_key, json)
        else:
            ret = await common_request(model_config.base_url, api_key, json)

        return ret

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


async def common_request(base_url: str, api_key: str, json: dict):
    timeout_config = config.timeout
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=json,
            timeout=(timeout_config if isinstance(timeout_config, int) else timeout_config.api_request),
        )
    if error := response.json().get("error"):
        raise RequestException(error["message"])
    return ChatCompletions(**response.json())


async def stream_request(base_url: str, api_key: str, json: dict):
    json["stream"] = True
    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=json,
        ) as response:
            ret_list: Optional[StreamChoiceList] = None
            async for chunk in response.aiter_lines():
                ret = sse_middle(chunk)
                if ret is None:
                    continue
                elif ret[0] == "data":
                    if ret[1] == "[DONE]":
                        break
                    else:
                        try:
                            i = loads(ret[1])
                            if ret_list is None:
                                ret_list = StreamChoiceList(**i)
                            else:
                                ret_list += StreamChoiceList(**i)
                        except Exception as e:
                            logger.error(f"解析数据块失败：{ret[1]} ||{e}")

                elif ret[0] == "::":
                    logger.debug(f"收到SSE注释：{ret[1]}")
                    continue
                elif ret[0] == "error":
                    raise RequestException(ret[1])
                else:
                    continue
            if ret_list is None:
                raise RequestException("Oops! 网络超时，请稍后重试")
            return ret_list.transform()


def sse_middle(
    line: str,
) -> Union[tuple[Literal["data", "event", "id", "retry", "::", "error"], str], None]:
    """单行SSE数据解析"""
    line = line.strip("\r")
    if not line:
        return None
    if ":" in line:
        field, value = line.split(":", 1)
        value = value.strip()
    else:
        return None
    if field == "":
        return "::", value
    elif field == "data" or field == "event" or field == "id" or field == "retry":
        return field, value

    return "error", line
