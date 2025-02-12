from typing import Optional


class TTSResponse:
    def __init__(self, model: str, speakers: Optional[list[str]] = None) -> None:
        self.model: str = model
        self.speakers: list[str] = speakers if speakers else []

    @classmethod
    async def create(cls, model: str) -> "TTSResponse":
        """异步创建 TTSResponse 对象"""
        from ..apis.request import API

        speakers = await API.get_tts_speakers(model)
        return cls(model, speakers)
