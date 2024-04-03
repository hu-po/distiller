"""
Anthropic Claude Wrapper
https://console.anthropic.com/docs/api/reference
"""

import os
import anthropic
import base64
import logging

log = logging.getLogger(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def list_models() -> list[str]:
    return ["claude-3-opus-20240229", "claude-3-haiku-20240229"]


def text(prompt: str, model: str = "claude-3-opus-20240229", max_tokens: int = 256):
    # https://docs.anthropic.com/claude/reference/messages_post
    response = client.messages.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
    )
    response = response.content
    log.info(
        f"\n---\nAnthropic {model}\n---\nprompt: {prompt}\nresponse: {response}\n---"
    )
    return response


def image(
    prompt: str,
    image_path: str,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 256,
) -> str:
    # https://docs.anthropic.com/claude/reference/messages-examples#vision
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    response = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ],
    )
    response = response.content[0].text
    log.info(
        f"\n---\nAnthropic {model}\n---\nprompt: {prompt}\nresponse: {response}\n---"
    )
    return response


async def async_image(*args, **kwargs) -> str:
    return image(*args, **kwargs)


async def async_text(*args, **kwargs) -> str:
    return text(*args, **kwargs)


def test():
    log.debug("Testing Anthropic Claude Wrapper")
    for model in list_models():
        log.debug(model)
    text("What is your name")
    image("Describe the image", "cover.png")


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    test()
