import re


def test_extract_content_and_think():
    from nonebot_plugin_deepseek.schemas import Message

    def extract_content_and_think(message: Message) -> tuple[str, str]:
        thinking = message.reasoning_content

        if not thinking:
            think_blocks = re.findall(r"<think>(.*?)</think>", message.content or "", flags=re.DOTALL)
            thinking = "\n".join([block.strip() for block in think_blocks if block.strip()])

        content = re.sub(r"<think>.*?</think>", "", message.content or "", flags=re.DOTALL).strip()

        return content, thinking

    # 没有 <think> 标签时
    message1 = Message(role="assistant", content="This is the response without think tags.", reasoning_content=None)
    clean_content, thinking = extract_content_and_think(message1)
    assert clean_content == "This is the response without think tags."
    assert thinking == ""

    # 有 reasoning_content 字段时
    message3 = Message(
        role="assistant", content="This is the response.", reasoning_content="reasoning content provided"
    )
    clean_content, thinking = extract_content_and_think(message3)
    assert clean_content == "This is the response."
    assert thinking == "reasoning content provided"

    # 有 <think> 标签时，无 reasoning_content 字段
    message2 = Message(
        role="assistant", content="<think>thinking part</think>This is the response.", reasoning_content=None
    )
    clean_content, thinking = extract_content_and_think(message2)
    assert clean_content == "This is the response."
    assert thinking == "thinking part"
