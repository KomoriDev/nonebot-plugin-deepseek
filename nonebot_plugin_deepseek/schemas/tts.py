from clilte.core import dataclass


@dataclass
class TTSModelInfo:
    model_name: str
    language_emotions: dict[str, list[str]]
