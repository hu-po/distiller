"""
OpenAI GPT Wrapper
https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
"""

import os
import openai
from openai import OpenAI
import base64
import requests
import logging

log = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def list_models() -> list[str]:
    return [m.id for m in openai.models.list()]


def text(prompt: str, model: str = "gpt-4-1106-preview", max_tokens: int = 256) -> str:
    # https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
    )
    response = response.choices[0].message.content
    log.info(f"\n---\nOpenAI {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
    return response


def image(
    prompt: str,
    image_path: str,
    model: str = "gpt-4-vision-preview",
    max_tokens: int = 256,
) -> str:
    # https://platform.openai.com/docs/guides/vision
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()["choices"][0]["message"]["content"]
    log.info(f"\n---\nOpenAI {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
    return response


async def async_image(*args, **kwargs) -> str:
    return image(*args, **kwargs)


async def async_text(*args, **kwargs) -> str:
    return text(*args, **kwargs)


def test():
    log.debug("Testing OpenAI GPT Wrapper")
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
