"""
Google Gemini Wrapper
https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
"""

import os
import google.generativeai as genai
import PIL.Image
import logging

log = logging.getLogger(__name__)
genai.configure(api_key=os.environ["GOOGLE_SDK_API_KEY"])


def list_models() -> list[str]:
    return [m.name for m in genai.list_models()]


def text(prompt: str, model: str = "models/gemini-pro"):
    # https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    response = genai.GenerativeModel(model).generate_content(prompt).text
    log.info(f"\n---\nGoogle {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
    return response


def image(
    prompt: str,
    image_path: str,
    model: str = "models/gemini-pro-vision",
) -> str:
    # https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    response = (
        genai.GenerativeModel(model)
        .generate_content([prompt, PIL.Image.open(image_path)])
        .text
    )
    log.info(f"\n---\nGoogle {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
    return response


async def async_image(*args, **kwargs) -> str:
    return image(*args, **kwargs)


async def async_text(*args, **kwargs) -> str:
    return text(*args, **kwargs)


def test():
    log.debug("Testing Google Gemini Wrapper")
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
