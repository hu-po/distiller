import asyncio
import logging

log = logging.getLogger(__name__)


def caption(
    image_path: str = "cover.png",
    prompt: str = "Describe the image",
    apis: list[str] = ["claude", "openai"],
) -> dict[str, str]:
    assert all(
        api in ["claude", "openai", "gemini", "mistral"] for api in apis
    ), f"Invalid API {apis}"
    tasks = []
    if "claude" in apis:
        import api_claude

        tasks.append(
            asyncio.create_task(api_claude.image(prompt, image_path), name="claude")
        )
    if "openai" in apis:
        import api_openai

        tasks.append(
            asyncio.create_task(api_openai.image(prompt, image_path), name="openai")
        )
    if "gemini" in apis:
        import api_gemini

        tasks.append(
            asyncio.create_task(api_gemini.image(prompt, image_path), name="gemini")
        )
    if "mistral" in apis:
        import api_mistral

        tasks.append(
            asyncio.create_task(api_mistral.image(prompt, image_path), name="mistral")
        )
    results = asyncio.gather(*tasks)
    results_dict = {}
    for task, result in zip(tasks, results):
        results_dict[task.get_name()] = result
    log.debug(results_dict)
    return results_dict


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    caption()
