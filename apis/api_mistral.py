"""
Mistral via Replicate API
"""

import replicate
import base64
import logging

log = logging.getLogger(__name__)


def list_models() -> list[str]:
    return [
        "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
        "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
    ]


def text(
    prompt: str,
    model: str = "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
    max_tokens: int = 256,
) -> str:
    # https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1?input=python
    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
        },
    )
    response = "".join(output)
    log.info(
        f"\n---\nReplicate {model}\n---\nprompt: {prompt}\nresponse: {response}\n---"
    )
    return response


def image(
    prompt: str,
    image_path: str,
    model: str = "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
    max_tokens: int = 256,
) -> str:
    # https://replicate.com/yorickvp/llava-v1.6-34b?input=python
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    output = replicate.run(
        model,
        input={
            "image": f"data:image/jpeg;base64,{base64_image}",
            "prompt": prompt,
            "max_tokens": max_tokens,
        },
    )
    response = "".join(output)
    log.info(
        f"\n---\nReplicate {model}\n---\nprompt: {prompt}\nresponse: {response}\n---"
    )
    return response


async def async_image(*args, **kwargs) -> str:
    return image(*args, **kwargs)


async def async_text(*args, **kwargs) -> str:
    return text(*args, **kwargs)


def test():
    log.debug("Testing Mistral via Replicate API")
    for model in list_models():
        log.debug(model)
    text("What is your name?")
    image("Describe the image", "cover.png")


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    test()
